from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import and_, case, func, or_, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from memoryos.memory.ranking import rerank
from memoryos.models import Memory, MemoryStatus
from memoryos.schemas.memory import MemoryResponse
from memoryos.schemas.retrieval import MemoryRetrievalResult, RetrievedMemory

SEMANTIC_CANDIDATE_LIMIT = 20


class RetrievalError(Exception):
    pass


def eligible_memories(user_id: UUID, now: datetime):
    return and_(
        Memory.user_id == user_id,
        Memory.status == MemoryStatus.ACTIVE,
        Memory.superseded_by.is_(None),
        or_(Memory.expires_at.is_(None), Memory.expires_at > now),
    )


def candidate_query(user_id: UUID, vector: list[float], now: datetime):
    # CASE prevents the planner evaluating cosine distance on incompatible vectors.
    distance = case(
        (
            and_(
                func.vector_dims(Memory.embedding) == len(vector),
                func.vector_norm(Memory.embedding) > 0,
            ),
            Memory.embedding.cosine_distance(vector),
        ),
        else_=None,
    ).label("cosine_distance")
    return (
        select(Memory, distance)
        .where(
            eligible_memories(user_id, now), Memory.embedding.is_not(None), distance.is_not(None)
        )
        .order_by(distance.asc(), Memory.id.asc())
        .limit(SEMANTIC_CANDIDATE_LIMIT)
    )


def select_candidates(db: Session, user_id: UUID, vector: list[float], now: datetime):
    return [
        (memory, distance) for memory, distance in db.execute(candidate_query(user_id, vector, now))
    ]


def retrieve_memories(
    db: Session,
    user_id: UUID,
    vector: list[float],
    limit: int = 5,
    debug: bool = False,
    *,
    now: datetime | None = None,
) -> MemoryRetrievalResult:
    if not 1 <= limit <= SEMANTIC_CANDIDATE_LIMIT:
        raise ValueError("Retrieval limit must be between 1 and 20")
    now = now or datetime.now(UTC)
    try:
        ranked = rerank(select_candidates(db, user_id, vector, now), now)[:limit]
        if not ranked:
            return MemoryRetrievalResult(memories=[])

        # Increment in SQL so simultaneous retrievals cannot overwrite each other's counts.
        statement = (
            update(Memory)
            .where(
                eligible_memories(user_id, now),
                Memory.id.in_([item.memory.id for item in ranked]),
            )
            .values(
                access_count=Memory.access_count + 1,
                last_accessed_at=case(
                    (Memory.last_accessed_at > now, Memory.last_accessed_at), else_=now
                ),
                updated_at=Memory.updated_at,
            )
            .returning(Memory)
            .execution_options(populate_existing=True, synchronize_session=False)
        )
        updated = {memory.id: memory for memory in db.scalars(statement)}
        results = [
            RetrievedMemory(
                memory=MemoryResponse.model_validate(updated[item.memory.id]),
                score=item.score,
                **({"components": item.components} if debug else {}),
            )
            for item in ranked
            if item.memory.id in updated
        ]
        db.commit()
        return MemoryRetrievalResult(memories=results)
    except SQLAlchemyError as exc:
        db.rollback()
        raise RetrievalError from exc
