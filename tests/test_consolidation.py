import json
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from itertools import combinations
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from memoryos.api.consolidation import get_consolidation_service
from memoryos.config import Settings
from memoryos.db.session import get_db
from memoryos.main import create_app
from memoryos.memory.clustering import find_clusters
from memoryos.memory.prompts import CONSOLIDATION_PROMPT
from memoryos.models import ConsolidationSource, Memory, MemoryType
from memoryos.schemas.consolidation import ConsolidationProposal
from memoryos.services.consolidation import MemoryConsolidationService, similarity_query
from memoryos.services.embeddings import EmbeddingService
from memoryos.services.ollama import APIConnectionError, APITimeoutError, RateLimitError


def proposal(**overrides):
    return {
        "content": "The user regularly takes a morning walk near home.",
        "confidence": 0.9,
        "should_consolidate": True,
        "reason": "The episodes consistently describe this routine.",
        **overrides,
    }


def response(**overrides):
    return SimpleNamespace(status="completed", output=[], output_parsed=proposal(**overrides))


@pytest.fixture
def consolidation_llm():
    client = Mock()
    client.parse = AsyncMock(return_value=response())
    client.embed = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])
    )
    return client


@pytest.fixture
def consolidation_client(memory_client, consolidation_llm):
    service = MemoryConsolidationService(
        consolidation_llm,
        EmbeddingService(consolidation_llm, "test-embedding"),
        model="test-consolidation",
    )
    memory_client.app.dependency_overrides[get_consolidation_service] = lambda: service
    return memory_client


def seed(client, user_id, **overrides):
    with contextmanager(client.app.dependency_overrides[get_db])() as db:
        memory = Memory(
            **{
                "id": uuid4(),
                "user_id": user_id,
                "content": "I took a morning walk near home.",
                "memory_type": MemoryType.EPISODIC,
                "embedding": [1, 0],
                "confidence": 0.9,
                "importance": 0.6,
                "metadata_": {"place": "home"},
                "source": "conversation",
                **overrides,
            }
        )
        db.add(memory)
        db.commit()
        return str(memory.id)


def seed_cluster(client, user_id, **overrides):
    return [seed(client, user_id, **overrides) for _ in range(3)]


def consolidate(client, user_id, **overrides):
    return client.post("/memory/consolidate", json={"user_id": str(user_id), **overrides})


def assert_no_consolidation(client, source_ids):
    records = client.get("/memories").json()
    assert all(item["consolidated_at"] is None for item in records)
    assert not any(item["source"] == "consolidation" for item in records)
    assert set(source_ids).issubset({item["id"] for item in records})
    with contextmanager(client.app.dependency_overrides[get_db])() as db:
        assert db.scalar(select(func.count()).select_from(ConsolidationSource)) == 0


def test_consolidates_with_provenance_and_preserves_sources(
    consolidation_client, consolidation_llm
):
    user_id = uuid4()
    source_ids = seed_cluster(consolidation_client, user_id)
    before = {item: consolidation_client.get(f"/memories/{item}").json() for item in source_ids}
    result = consolidate(consolidation_client, user_id)
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["candidates_examined"] == 3
    assert len(data["memories"]) == 1
    consolidated = data["memories"][0]
    summary = consolidated["memory"]
    assert summary["memory_type"] == "SEMANTIC"
    assert summary["content"] == proposal()["content"]
    assert summary["importance"] == pytest.approx(0.6)
    assert summary["confidence"] == 0.9
    assert summary["embedding"] == [1, 0]
    assert set(consolidated["source_memory_ids"]) == set(source_ids)
    assert set(summary["metadata"]["source_memory_ids"]) == set(source_ids)
    assert summary["consolidated_at"] is None
    for source_id in source_ids:
        source = consolidation_client.get(f"/memories/{source_id}").json()
        assert source["consolidated_at"] is not None
        for field in (
            "content",
            "status",
            "confidence",
            "importance",
            "metadata",
            "embedding",
            "access_count",
            "last_accessed_at",
        ):
            assert source[field] == before[source_id][field]
    with contextmanager(consolidation_client.app.dependency_overrides[get_db])() as db:
        links = list(db.scalars(select(ConsolidationSource)))
        assert {str(link.source_memory_id) for link in links} == set(source_ids)
        assert {str(link.consolidated_memory_id) for link in links} == {summary["id"]}
    assert len(consolidation_client.get("/memories").json()) == 4
    call = consolidation_llm.parse.call_args.kwargs
    assert call["text_format"] is ConsolidationProposal
    assert call["model"] == "test-consolidation"
    assert call["input"][0]["content"] == CONSOLIDATION_PROMPT
    assert json.loads(call["input"][1]["content"])[0]["metadata"] == {"place": "home"}


@pytest.mark.parametrize("count", [0, 1, 2])
def test_requires_three_sources(consolidation_client, consolidation_llm, count):
    user_id = uuid4()
    ids = [seed(consolidation_client, user_id) for _ in range(count)]
    assert consolidate(consolidation_client, user_id).json()["memories"] == []
    consolidation_llm.parse.assert_not_awaited()
    consolidation_llm.embed.assert_not_awaited()
    assert_no_consolidation(consolidation_client, ids)


def test_insufficient_source_confidence_blocks_cluster(consolidation_client, consolidation_llm):
    user_id = uuid4()
    ids = [seed(consolidation_client, user_id, confidence=value) for value in (0.9, 0.9, 0.79)]
    assert consolidate(consolidation_client, user_id).json()["memories"] == []
    consolidation_llm.parse.assert_not_awaited()
    assert_no_consolidation(consolidation_client, ids)


@pytest.mark.parametrize("overrides", [{"confidence": 0.79}, {"should_consolidate": False}])
def test_llm_gate_leaves_sources_unmarked(consolidation_client, consolidation_llm, overrides):
    user_id = uuid4()
    ids = seed_cluster(consolidation_client, user_id)
    consolidation_llm.parse.return_value = response(**overrides)
    result = consolidate(consolidation_client, user_id).json()
    assert result["memories"] == []
    assert result["clusters_skipped"] == 1
    consolidation_llm.embed.assert_not_awaited()
    assert_no_consolidation(consolidation_client, ids)


def test_confidence_is_capped_by_sources_and_expiration_is_inherited(
    consolidation_client, consolidation_llm
):
    user_id = uuid4()
    expiry = datetime.now(UTC) + timedelta(days=3)
    seed(consolidation_client, user_id, confidence=0.8, expires_at=expiry)
    seed_cluster(consolidation_client, user_id)
    consolidation_llm.parse.return_value = response(confidence=1.0)
    summary = consolidate(consolidation_client, user_id).json()["memories"][0]["memory"]
    assert summary["confidence"] == 0.8
    assert (
        datetime.fromisoformat(summary["expires_at"].replace("Z", "+00:00")).replace(tzinfo=UTC)
        == expiry
    )


def test_repeated_runs_do_not_reuse_sources(consolidation_client, consolidation_llm):
    user_id = uuid4()
    seed_cluster(consolidation_client, user_id)
    assert len(consolidate(consolidation_client, user_id).json()["memories"]) == 1
    assert consolidate(consolidation_client, user_id).json()["memories"] == []
    assert len(consolidation_client.get("/memories").json()) == 4
    consolidation_llm.parse.assert_awaited_once()


def test_two_distinct_clusters_create_two_summaries(consolidation_client):
    user_id = uuid4()
    first = set(seed_cluster(consolidation_client, user_id, embedding=[1, 0]))
    second = set(seed_cluster(consolidation_client, user_id, embedding=[0, 1]))
    results = consolidate(consolidation_client, user_id).json()["memories"]
    assert len(results) == 2
    assert {frozenset(item["source_memory_ids"]) for item in results} == {
        frozenset(first),
        frozenset(second),
    }


def test_similarity_chains_are_not_clusters(consolidation_client, consolidation_llm):
    user_id = uuid4()
    # Adjacent vectors have cosine 0.866; the endpoints have cosine 0.5.
    ids = [
        seed(consolidation_client, user_id, embedding=vector)
        for vector in ([1, 0], [0.8660254, 0.5], [0.5, 0.8660254])
    ]
    assert consolidate(consolidation_client, user_id).json()["memories"] == []
    consolidation_llm.parse.assert_not_awaited()
    assert_no_consolidation(consolidation_client, ids)


def test_candidate_filtering_and_user_isolation(consolidation_client):
    user_id = uuid4()
    expected = seed_cluster(consolidation_client, user_id, status="STALE")
    excluded = seed_cluster(consolidation_client, uuid4())
    for fields in (
        {"memory_type": "SEMANTIC"},
        {"memory_type": "PREFERENCE"},
        {"status": "ARCHIVED"},
        {"status": "SUPERSEDED"},
        {"expires_at": datetime.now(UTC) - timedelta(days=1)},
        {"embedding": None},
        {"embedding": [0, 0]},
        {"confidence": 0.7},
        {"superseded_by": UUID(expected[0])},
    ):
        excluded.extend(seed_cluster(consolidation_client, user_id, **fields))
    result = consolidate(consolidation_client, user_id).json()
    assert result["candidates_examined"] == 3
    assert set(result["memories"][0]["source_memory_ids"]) == set(expected)
    for source_id in excluded:
        assert consolidation_client.get(f"/memories/{source_id}").json()["consolidated_at"] is None


def test_mixed_vector_dimensions_do_not_form_cluster(consolidation_client, consolidation_llm):
    user_id = uuid4()
    for vector in ([1], [1, 0], [1, 0, 0]):
        seed(consolidation_client, user_id, embedding=vector)
    assert consolidate(consolidation_client, user_id).json()["memories"] == []
    consolidation_llm.parse.assert_not_awaited()


def test_provenance_prevents_source_and_summary_deletion(consolidation_client):
    user_id = uuid4()
    ids = seed_cluster(consolidation_client, user_id)
    summary = consolidate(consolidation_client, user_id).json()["memories"][0]["memory"]
    for memory_id in [*ids, summary["id"]]:
        assert consolidation_client.delete(f"/memories/{memory_id}").status_code == 409
        assert consolidation_client.get(f"/memories/{memory_id}").status_code == 200


@pytest.mark.parametrize(
    "overrides",
    [
        {"content": " "},
        {"content": "x" * 501},
        {"confidence": 1.01},
        {"confidence": float("nan")},
        {"should_consolidate": "yes"},
        {"source_ids": []},
    ],
)
def test_invalid_structured_output_rolls_back(consolidation_client, consolidation_llm, overrides):
    user_id = uuid4()
    ids = seed_cluster(consolidation_client, user_id)
    consolidation_llm.parse.return_value = response(**overrides)
    assert consolidate(consolidation_client, user_id).status_code == 502
    assert_no_consolidation(consolidation_client, ids)


@pytest.mark.parametrize("kind, status", [("incomplete", 502), ("empty", 502), ("refusal", 422)])
def test_incomplete_and_refused_output(consolidation_client, consolidation_llm, kind, status):
    user_id = uuid4()
    ids = seed_cluster(consolidation_client, user_id)
    result = response()
    result.output_parsed = None
    if kind == "incomplete":
        result.status = "incomplete"
    if kind == "refusal":
        result.output = [SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal")])]
    consolidation_llm.parse.return_value = result
    assert consolidate(consolidation_client, user_id).status_code == status
    assert_no_consolidation(consolidation_client, ids)


@pytest.mark.parametrize(
    "error, status",
    [
        (APITimeoutError(), 504),
        (APIConnectionError(), 502),
        (
            RateLimitError(
                "private",
            ),
            503,
        ),
    ],
)
def test_second_cluster_failure_rolls_back_first_cluster(
    consolidation_client, consolidation_llm, error, status
):
    user_id = uuid4()
    ids = seed_cluster(consolidation_client, user_id) + seed_cluster(
        consolidation_client, user_id, embedding=[0, 1]
    )
    consolidation_llm.parse.side_effect = [response(), error]
    result = consolidate(consolidation_client, user_id)
    assert result.status_code == status
    assert "private" not in result.text
    assert_no_consolidation(consolidation_client, ids)


def test_embedding_failure_does_not_mark_sources(consolidation_client, consolidation_llm):
    user_id = uuid4()
    ids = seed_cluster(consolidation_client, user_id)
    consolidation_llm.embed.side_effect = APITimeoutError(
    )
    assert consolidate(consolidation_client, user_id).status_code == 504
    assert_no_consolidation(consolidation_client, ids)


def test_commit_failure_rolls_back_everything(consolidation_client, monkeypatch):
    user_id = uuid4()
    ids = seed_cluster(consolidation_client, user_id)
    with monkeypatch.context() as patch:
        patch.setattr(
            Session, "commit", Mock(side_effect=OperationalError("commit", {}, Exception()))
        )
        assert consolidate(consolidation_client, user_id).status_code == 503
    assert_no_consolidation(consolidation_client, ids)


def test_locked_user_is_retryable(consolidation_client, consolidation_llm, monkeypatch):
    monkeypatch.setattr("memoryos.services.consolidation.try_lock_user", lambda *args: False)
    assert consolidate(consolidation_client, uuid4()).status_code == 409
    consolidation_llm.parse.assert_not_awaited()


def test_pagination(consolidation_client, monkeypatch):
    monkeypatch.setattr("memoryos.services.consolidation.CANDIDATE_LIMIT", 3)
    user_id = uuid4()
    for index in range(6):
        seed(consolidation_client, user_id, id=UUID(int=index + 1))
    first = consolidate(consolidation_client, user_id).json()
    assert first["next_cursor"] == str(UUID(int=3))
    assert first["candidates_examined"] == 3
    second = consolidate(consolidation_client, user_id, after_id=first["next_cursor"]).json()
    assert second["next_cursor"] is None
    assert len(second["memories"]) == 1


def test_cluster_algorithm_is_deterministic_and_bounded():
    ids = [UUID(int=i) for i in range(1, 25)]
    pairs = {frozenset(pair) for pair in combinations(ids, 2)}
    forward = find_clusters(ids, pairs)
    assert forward == find_clusters(ids[::-1], pairs)
    assert [len(cluster) for cluster in forward] == [20, 4]
    assert len({memory_id for cluster in forward for memory_id in cluster}) == 24


def test_postgres_uses_guarded_cosine_comparisons():
    compiled = similarity_query([uuid4(), uuid4()], 0.85).compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert "<=>" in sql
    assert "CASE WHEN" in sql
    assert "vector_dims" in sql


def test_missing_configuration():
    with TestClient(create_app(Settings(_env_file=None))) as client:
        unavailable = client.app.state.llm_client
        client.app.state.llm_client = None
        assert consolidate(client, uuid4()).status_code == 503
        client.app.state.llm_client = unavailable
