from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from memoryos.memory.decay import evaluate_decay
from memoryos.models import Memory, MemoryStatus


@dataclass
class DecaySummary:
    evaluated_at: datetime
    dry_run: bool
    processed: int = 0
    stale: int = 0
    archived: int = 0
    unchanged: int = 0


def decay_memories(
    db: Session,
    *,
    now: datetime | None = None,
    batch_size: int = 500,
    dry_run: bool = False,
    user_id: UUID | None = None,
) -> DecaySummary:
    """Own a dedicated session; commit each batch and roll back a failing batch."""
    now = now or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Decay evaluation requires a timezone-aware timestamp")
    if not 1 <= batch_size <= 10000:
        raise ValueError("Batch size must be between 1 and 10000")
    summary = DecaySummary(evaluated_at=now, dry_run=dry_run)
    cursor = None
    try:
        while True:
            statement = (
                select(
                    Memory.id,
                    Memory.memory_type,
                    Memory.importance,
                    Memory.created_at,
                    Memory.last_accessed_at,
                    Memory.expires_at,
                    Memory.status,
                )
                .where(
                    Memory.status.in_([MemoryStatus.ACTIVE, MemoryStatus.STALE]),
                    Memory.superseded_by.is_(None),
                )
                .order_by(Memory.id)
                .limit(batch_size)
                .with_for_update(skip_locked=True)
            )
            if cursor is not None:
                statement = statement.where(Memory.id > cursor)
            if user_id is not None:
                statement = statement.where(Memory.user_id == user_id)
            rows = db.execute(statement).all()
            if not rows:
                db.rollback()
                break
            for row in rows:
                decision = evaluate_decay(
                    memory_type=row.memory_type,
                    importance=row.importance,
                    created_at=row.created_at,
                    last_accessed_at=row.last_accessed_at,
                    expires_at=row.expires_at,
                    status=row.status,
                    now=now,
                )
                summary.processed += 1
                if decision.status == row.status:
                    summary.unchanged += 1
                    continue
                if decision.status == MemoryStatus.ARCHIVED:
                    summary.archived += 1
                else:
                    summary.stale += 1
                if not dry_run:
                    db.execute(
                        update(Memory)
                        .where(Memory.id == row.id)
                        .values(status=decision.status, updated_at=now)
                        .execution_options(synchronize_session=False)
                    )
            cursor = rows[-1].id
            if dry_run:
                db.rollback()
            else:
                db.commit()
    except Exception:
        db.rollback()
        raise
    return summary
