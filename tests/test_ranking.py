from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest

from memoryos.memory.ranking import rerank
from memoryos.models import Memory

NOW = datetime(2026, 9, 22, tzinfo=UTC)


def memory(index=1, **overrides):
    return Memory(
        id=UUID(int=index),
        **{"created_at": NOW, "importance": 0.5, "confidence": 0.8, "access_count": 0, **overrides},
    )


def test_exact_weighted_score():
    target = memory(created_at=NOW - timedelta(days=30), access_count=5)
    other = memory(2, access_count=10)
    ranked = rerank([(target, 0.2), (other, 1.0)], NOW)
    result = next(item for item in ranked if item.memory.id == target.id)
    assert result.components.model_dump() == pytest.approx(
        {
            "semantic_similarity": 0.8,
            "importance": 0.5,
            "recency": 0.5,
            "access_frequency": 0.5,
            "confidence": 0.8,
        }
    )
    assert result.score == pytest.approx(
        0.50 * 0.8 + 0.20 * 0.5 + 0.15 * 0.5 + 0.10 * 0.5 + 0.05 * 0.8
    )


def test_reranking_can_change_semantic_order():
    similar = memory(1, importance=0, confidence=0, created_at=NOW - timedelta(days=365))
    useful = memory(2, importance=1, confidence=1, access_count=10)
    assert [item.memory.id for item in rerank([(similar, 0), (useful, 0.2)], NOW)] == [
        useful.id,
        similar.id,
    ]


def test_zero_access_empty_candidates_and_deterministic_ties():
    assert rerank([], NOW) == []
    results = rerank([(memory(2), 0.2), (memory(1), 0.2)], NOW)
    assert [item.memory.id.int for item in results] == [1, 2]
    assert all(item.components.access_frequency == 0 for item in results)


@pytest.mark.parametrize("days, recency", [(0, 1), (30, 0.5), (60, 0.25), (-1, 1)])
def test_recency(days, recency):
    result = rerank([(memory(created_at=NOW - timedelta(days=days)), 0.0)], NOW)[0]
    assert result.components.recency == pytest.approx(recency)


def test_negative_cosine_is_preserved_and_nonfinite_distance_is_ignored():
    results = rerank([(memory(1), 2.0), (memory(2), float("nan"))], NOW)
    assert len(results) == 1
    assert results[0].components.semantic_similarity == -1
