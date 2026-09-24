from dataclasses import dataclass
from datetime import UTC, datetime

from memoryos.models import MemoryStatus, MemoryType

HALF_LIFE_DAYS = {
    MemoryType.WORKING: 0.25,
    MemoryType.TASK: 7.0,
    MemoryType.EPISODIC: 30.0,
    MemoryType.SEMANTIC: 180.0,
    MemoryType.PREFERENCE: 365.0,
}
ACTIVE_THRESHOLD = 0.20
STALE_THRESHOLD = 0.05


def as_utc(value: datetime) -> datetime:
    # PostgreSQL timestamps are aware; SQLite test timestamps are naive UTC.
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


@dataclass(frozen=True)
class DecayDecision:
    score: float
    inactive_days: float
    status: MemoryStatus


def evaluate_decay(
    *,
    memory_type: MemoryType,
    importance: float,
    created_at: datetime,
    last_accessed_at: datetime | None,
    status: MemoryStatus,
    expires_at: datetime | None,
    now: datetime,
) -> DecayDecision:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("Decay evaluation requires a timezone-aware timestamp")
    if not 0 <= importance <= 1:
        raise ValueError("Importance must be a finite number between zero and one")
    inactive_days = max(0.0, (now - as_utc(last_accessed_at or created_at)).total_seconds() / 86400)
    score = importance * 2 ** (-inactive_days / HALF_LIFE_DAYS[memory_type])
    if status in (MemoryStatus.ARCHIVED, MemoryStatus.SUPERSEDED):
        target = status
    elif (expires_at is not None and as_utc(expires_at) <= now) or score < STALE_THRESHOLD:
        target = MemoryStatus.ARCHIVED
    elif score < ACTIVE_THRESHOLD or status == MemoryStatus.STALE:
        target = MemoryStatus.STALE
    else:
        target = MemoryStatus.ACTIVE
    return DecayDecision(score=score, inactive_days=inactive_days, status=target)
