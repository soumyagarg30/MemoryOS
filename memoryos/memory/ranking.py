import math
from dataclasses import dataclass
from datetime import UTC, datetime

from memoryos.models import Memory
from memoryos.schemas.retrieval import RetrievalComponents

RECENCY_HALF_LIFE_DAYS = 30.0


@dataclass(frozen=True)
class RankedMemory:
    memory: Memory
    score: float
    components: RetrievalComponents


def rerank(candidates: list[tuple[Memory, float]], now: datetime) -> list[RankedMemory]:
    max_access_count = max((memory.access_count for memory, _ in candidates), default=0)
    ranked = []
    for memory, distance in candidates:
        if not math.isfinite(distance):
            continue
        created_at = memory.created_at
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        age_days = max(0.0, (now - created_at).total_seconds() / 86400)
        components = RetrievalComponents(
            semantic_similarity=max(-1.0, min(1.0, 1.0 - distance)),
            importance=memory.importance,
            recency=2 ** (-age_days / RECENCY_HALF_LIFE_DAYS),
            access_frequency=memory.access_count / max_access_count if max_access_count else 0.0,
            confidence=memory.confidence,
        )
        score = (
            0.50 * components.semantic_similarity
            + 0.20 * components.importance
            + 0.15 * components.recency
            + 0.10 * components.access_frequency
            + 0.05 * components.confidence
        )
        ranked.append(RankedMemory(memory, score, components))
    return sorted(ranked, key=lambda item: (-item.score, str(item.memory.id)))
