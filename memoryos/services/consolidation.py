import json
from datetime import UTC, datetime
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import and_, case, exists, func, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, aliased
from starlette.concurrency import run_in_threadpool

from memoryos.db.locks import try_lock_user
from memoryos.memory.clustering import find_clusters
from memoryos.memory.decay import as_utc
from memoryos.memory.prompts import CONSOLIDATION_PROMPT
from memoryos.models import ConsolidationSource, Memory, MemoryStatus, MemoryType
from memoryos.schemas.consolidation import (
    ConsolidatedMemory,
    ConsolidationProposal,
    ConsolidationResult,
)
from memoryos.schemas.memory import MemoryResponse
from memoryos.services.embeddings import EmbeddingService
from memoryos.services.ollama import OllamaClient

CANDIDATE_LIMIT = 100


class ConsolidationOutputError(Exception):
    pass


class ConsolidationRefused(ConsolidationOutputError):
    pass


class ConsolidationBusy(Exception):
    pass


class ConsolidationDatabaseError(Exception):
    pass


def load_candidates(
    db: Session, user_id: UUID, min_confidence: float, after_id: UUID | None
) -> tuple[list[Memory], UUID | None]:
    if not try_lock_user(db, user_id):
        raise ConsolidationBusy
    now = datetime.now(UTC)
    statement = (
        select(Memory)
        .where(
            Memory.user_id == user_id,
            Memory.memory_type == MemoryType.EPISODIC,
            Memory.status.in_([MemoryStatus.ACTIVE, MemoryStatus.STALE]),
            Memory.superseded_by.is_(None),
            Memory.consolidated_at.is_(None),
            ~exists().where(ConsolidationSource.source_memory_id == Memory.id),
            Memory.confidence >= min_confidence,
            Memory.embedding.is_not(None),
            func.vector_norm(Memory.embedding) > 0,
            or_(Memory.expires_at.is_(None), Memory.expires_at > now),
        )
        .order_by(Memory.id)
        .limit(CANDIDATE_LIMIT + 1)
        .with_for_update()
    )
    if after_id is not None:
        statement = statement.where(Memory.id > after_id)
    candidates = list(db.scalars(statement))
    next_cursor = candidates[CANDIDATE_LIMIT - 1].id if len(candidates) > CANDIDATE_LIMIT else None
    return candidates[:CANDIDATE_LIMIT], next_cursor


def similarity_query(ids: list[UUID], threshold: float):
    left, right = aliased(Memory), aliased(Memory)
    distance = case(
        (
            and_(
                func.vector_dims(left.embedding) == func.vector_dims(right.embedding),
                func.vector_norm(left.embedding) > 0,
                func.vector_norm(right.embedding) > 0,
            ),
            left.embedding.cosine_distance(right.embedding),
        ),
        else_=None,
    )
    return (
        select(left.id, right.id)
        .join(right, left.id < right.id)
        .where(left.id.in_(ids), right.id.in_(ids), distance <= 1.0 - threshold)
    )


def cluster_candidates(
    db: Session, candidates: list[Memory], threshold: float
) -> list[list[Memory]]:
    if len(candidates) < 3:
        return []
    by_id = {memory.id: memory for memory in candidates}
    pairs = {
        frozenset((left, right))
        for left, right in db.execute(similarity_query(list(by_id), threshold))
    }
    return [
        [by_id[memory_id] for memory_id in group] for group in find_clusters(list(by_id), pairs)
    ]


def persist_cluster(
    db: Session,
    sources: list[Memory],
    proposal: ConsolidationProposal,
    embedding: list[float],
    model: str,
) -> ConsolidatedMemory:
    now = datetime.now(UTC)
    if any(
        memory.expires_at is not None and as_utc(memory.expires_at) <= now for memory in sources
    ):
        raise ConsolidationBusy
    expirations = [as_utc(memory.expires_at) for memory in sources if memory.expires_at is not None]
    source_ids = [memory.id for memory in sources]
    summary = Memory(
        user_id=sources[0].user_id,
        content=proposal.content,
        memory_type=MemoryType.SEMANTIC,
        embedding=embedding,
        importance=sum(memory.importance for memory in sources) / len(sources),
        confidence=min(proposal.confidence, *(memory.confidence for memory in sources)),
        expires_at=min(expirations) if expirations else None,
        source="consolidation",
        metadata_={
            "source_memory_ids": [str(item) for item in source_ids],
            "consolidation_model": model,
        },
    )
    db.add(summary)
    db.flush()
    for source in sources:
        source.consolidated_at = now
        db.add(
            ConsolidationSource(
                source_memory_id=source.id, consolidated_memory_id=summary.id, created_at=now
            )
        )
    db.flush()
    db.refresh(summary)
    return ConsolidatedMemory(
        memory=MemoryResponse.model_validate(summary), source_memory_ids=source_ids
    )


class MemoryConsolidationService:
    def __init__(
        self,
        client: OllamaClient,
        embeddings: EmbeddingService,
        *,
        model: str,
        similarity: float = 0.85,
        min_confidence: float = 0.8,
    ) -> None:
        if not 0 <= similarity <= 1 or not 0 <= min_confidence <= 1:
            raise ValueError("Similarity and confidence thresholds must be between zero and one")
        self.client = client
        self.embeddings = embeddings
        self.model = model
        self.similarity = similarity
        self.min_confidence = min_confidence

    async def summarize(self, sources: list[Memory]) -> ConsolidationProposal:
        context = [
            {
                "content": memory.content,
                "confidence": memory.confidence,
                "created_at": memory.created_at.isoformat(),
                "source": memory.source,
                "metadata": memory.metadata_,
            }
            for memory in sources
        ]
        try:
            response = await self.client.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": CONSOLIDATION_PROMPT},
                    {"role": "user", "content": json.dumps(context, allow_nan=False)},
                ],
                text_format=ConsolidationProposal,
                max_output_tokens=2048,
            )
            if response.status != "completed":
                raise ConsolidationOutputError
            if any(
                item.type == "message" and any(part.type == "refusal" for part in item.content)
                for item in response.output
            ):
                raise ConsolidationRefused
            return ConsolidationProposal.model_validate(response.output_parsed)
        except (ValidationError, ValueError) as exc:
            raise ConsolidationOutputError from exc

    async def consolidate(
        self, db: Session, user_id: UUID, after_id: UUID | None = None
    ) -> ConsolidationResult:
        try:
            candidates, cursor = await run_in_threadpool(
                load_candidates, db, user_id, self.min_confidence, after_id
            )
            clusters = await run_in_threadpool(cluster_candidates, db, candidates, self.similarity)
            result = ConsolidationResult(
                memories=[],
                candidates_examined=len(candidates),
                clusters_skipped=0,
                next_cursor=cursor,
            )
            for sources in clusters:
                proposal = await self.summarize(sources)
                if not proposal.should_consolidate or proposal.confidence < self.min_confidence:
                    result.clusters_skipped += 1
                    continue
                vector = await self.embeddings.embed(proposal.content)
                consolidated = await run_in_threadpool(
                    persist_cluster, db, sources, proposal, vector, self.model
                )
                result.memories.append(consolidated)
            await run_in_threadpool(db.commit)
            return result
        except SQLAlchemyError as exc:
            await run_in_threadpool(db.rollback)
            raise ConsolidationDatabaseError from exc
        except Exception:
            await run_in_threadpool(db.rollback)
            raise
