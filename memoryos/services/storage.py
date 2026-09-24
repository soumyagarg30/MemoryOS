import math
from dataclasses import dataclass
from datetime import UTC, datetime

from pgvector import Vector
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from memoryos.db.locks import try_lock_user
from memoryos.models import Memory, MemoryStatus
from memoryos.schemas.conflicts import MemoryComparison, MemoryConflictAnalysis, MemoryRelationship
from memoryos.schemas.memory import MemoryCreate, MemoryResponse
from memoryos.services.conflicts import MemoryConflictDetector
from memoryos.services.embeddings import EmbeddingService
from memoryos.services.memories import InvalidSupersedingMemory, MemoryConflict
from memoryos.services.retrieval import candidate_query

CONFIDENCE_INCREMENT = 0.05


class MemoryStorageError(Exception):
    pass


class MemoryWriteBusy(MemoryConflict):
    pass


class InvalidMemoryEmbedding(Exception):
    pass


@dataclass(frozen=True)
class MemoryWriteResult:
    memory: MemoryResponse
    created: bool
    comparisons: tuple[MemoryComparison, ...] = ()
    reinforced: tuple[MemoryResponse, ...] = ()
    superseded: tuple[MemoryResponse, ...] = ()


def is_current(payload: MemoryCreate, now: datetime) -> bool:
    return (
        payload.status == MemoryStatus.ACTIVE
        and payload.superseded_by is None
        and (payload.expires_at is None or payload.expires_at > now)
    )


def prepare_write(db: Session, payload: MemoryCreate) -> list[Memory]:
    # Also protects the empty candidate set from concurrent writes.
    if not try_lock_user(db, payload.user_id):
        raise MemoryWriteBusy
    if payload.superseded_by is not None:
        replacement = db.get(Memory, payload.superseded_by)
        if replacement is None or replacement.user_id != payload.user_id:
            raise InvalidSupersedingMemory
    now = datetime.now(UTC)
    if not is_current(payload, now):
        return []
    statement = candidate_query(payload.user_id, payload.embedding, now).with_for_update()
    return [memory for memory, _ in db.execute(statement)]


def apply_analysis(
    db: Session, payload: MemoryCreate, candidates: list[Memory], analysis: MemoryConflictAnalysis
) -> MemoryWriteResult:
    relationships = {item.memory_id: item.relationship for item in analysis.comparisons}
    now = datetime.now(UTC)
    if candidates and not is_current(payload, now):
        raise MemoryWriteBusy
    for memory in candidates:
        expires_at = memory.expires_at
        if (
            expires_at
            and (expires_at.replace(tzinfo=UTC) if expires_at.tzinfo is None else expires_at) <= now
        ):
            raise MemoryWriteBusy

    reinforced = [
        memory
        for memory in candidates
        if relationships[memory.id] == MemoryRelationship.REINFORCEMENT
    ]
    if reinforced:
        # Candidate order is semantic similarity, then UUID; choose a stable canonical record.
        target = reinforced[0]
        for memory in reinforced:
            memory.confidence = min(1.0, memory.confidence + CONFIDENCE_INCREMENT)
    else:
        target = Memory(**payload.model_dump(exclude={"metadata"}), metadata_=payload.metadata)
        db.add(target)
        db.flush()

    for memory in candidates:
        if relationships[memory.id] == MemoryRelationship.CONTRADICTION:
            memory.status = MemoryStatus.SUPERSEDED
            memory.superseded_by = target.id
    db.flush()
    db.refresh(target)
    result = MemoryWriteResult(
        memory=MemoryResponse.model_validate(target),
        created=not reinforced,
        comparisons=tuple(analysis.comparisons),
        reinforced=tuple(MemoryResponse.model_validate(m) for m in reinforced),
        superseded=tuple(
            MemoryResponse.model_validate(m)
            for m in candidates
            if relationships[m.id] == MemoryRelationship.CONTRADICTION
        ),
    )
    db.commit()
    return result


class MemoryStorageService:
    def __init__(self, embeddings: EmbeddingService, detector: MemoryConflictDetector) -> None:
        self.embeddings = embeddings
        self.detector = detector

    async def store(self, db: Session, payload: MemoryCreate) -> MemoryWriteResult:
        vector = payload.embedding
        if vector is None:
            vector = await self.embeddings.embed(payload.content)
        elif not math.hypot(*Vector(vector).to_list()):
            raise InvalidMemoryEmbedding
        payload = payload.model_copy(update={"embedding": vector})
        try:
            candidates = await run_in_threadpool(prepare_write, db, payload)
            analysis = await self.detector.classify(payload, candidates)
            return await run_in_threadpool(apply_analysis, db, payload, candidates, analysis)
        except IntegrityError as exc:
            await run_in_threadpool(db.rollback)
            raise MemoryConflict from exc
        except SQLAlchemyError as exc:
            await run_in_threadpool(db.rollback)
            raise MemoryStorageError from exc
        except Exception:
            await run_in_threadpool(db.rollback)
            raise
