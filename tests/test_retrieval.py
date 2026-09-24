import asyncio
import os
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from memoryos.api.memories import get_storage_service
from memoryos.api.retrieval import get_embedding_service
from memoryos.config import Settings
from memoryos.db.session import get_db
from memoryos.main import create_app
from memoryos.models import Base, Memory
from memoryos.services.embeddings import EmbeddingError, EmbeddingService
from memoryos.services.ollama import APIConnectionError, APITimeoutError, RateLimitError
from memoryos.services.retrieval import RetrievalError, candidate_query, retrieve_memories


@pytest.fixture
def mock_llm():
    client = Mock()
    client.embed = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])
    )
    return client


@pytest.fixture(
    params=["sqlite"] + (["postgres"] if os.environ.get("MEMORYOS_TEST_DATABASE_URL") else [])
)
def retrieval_client(request, mock_llm, storage_service):
    engine = None
    if request.param == "postgres":
        bind = request.getfixturevalue("postgres_connection")
        config = Config("alembic.ini")
        config.attributes.update(connection=bind, version_table_schema=bind.info["test_schema"])
        command.upgrade(config, "head")
    else:
        engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )

        Base.metadata.create_all(engine)
        bind = engine

    def override_db():
        with Session(bind, expire_on_commit=False, join_transaction_mode="create_savepoint") as db:
            yield db

    app = create_app(Settings(_env_file=None))
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_storage_service] = lambda: storage_service
    app.dependency_overrides[get_embedding_service] = lambda: EmbeddingService(
        mock_llm, "test-embedding"
    )
    try:
        with TestClient(app) as client:
            yield client
    finally:
        if engine:
            engine.dispose()


def create_memory(client, user_id, **overrides):
    response = client.post(
        "/memories",
        json={
            "user_id": str(user_id),
            "content": "A useful fact",
            "memory_type": "SEMANTIC",
            "embedding": [1.0, 0.0],
            **overrides,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def retrieve(client, user_id, **overrides):
    return client.post(
        "/memory/retrieve", json={"user_id": str(user_id), "query": "Useful facts", **overrides}
    )


def test_default_five_and_only_returned_memories_are_updated(retrieval_client, mock_llm):
    user_id = uuid4()
    original = [create_memory(retrieval_client, user_id, importance=i / 6) for i in range(7)]
    response = retrieve(retrieval_client, user_id)
    assert response.status_code == 200, response.text
    results = response.json()["memories"]
    assert len(results) == 5
    assert [item["memory"]["id"] for item in results] == [item["id"] for item in original[::-1][:5]]
    assert all("components" not in item for item in results)
    chosen = {item["memory"]["id"] for item in results}
    for item in original:
        stored = retrieval_client.get(f"/memories/{item['id']}").json()
        assert stored["access_count"] == (1 if item["id"] in chosen else 0)
        assert (stored["last_accessed_at"] is not None) == (item["id"] in chosen)
        assert stored["updated_at"] == item["updated_at"]
    assert all(item["memory"]["access_count"] == 1 for item in results)
    mock_llm.embed.assert_awaited_once_with(
        model="test-embedding", input="Useful facts"
    )


def test_debug_scores_precede_access_update(retrieval_client):
    user_id = uuid4()
    create_memory(retrieval_client, user_id)
    first = retrieve(retrieval_client, user_id, debug=True).json()["memories"][0]
    c = first["components"]
    assert c["access_frequency"] == 0
    assert first["score"] == pytest.approx(
        0.50 * c["semantic_similarity"]
        + 0.20 * c["importance"]
        + 0.15 * c["recency"]
        + 0.10 * c["access_frequency"]
        + 0.05 * c["confidence"]
    )
    second = retrieve(retrieval_client, user_id, debug=True).json()["memories"][0]
    assert second["components"]["access_frequency"] == 1
    assert second["memory"]["access_count"] == 2
    assert second["memory"]["last_accessed_at"] >= first["memory"]["last_accessed_at"]


def test_filters_and_incompatible_embeddings(retrieval_client):
    user_id = uuid4()
    active = create_memory(retrieval_client, user_id, memory_type="TASK")
    excluded = [create_memory(retrieval_client, uuid4())]
    for status in ("STALE", "ARCHIVED", "SUPERSEDED"):
        excluded.append(create_memory(retrieval_client, user_id, status=status))
    for fields in (
        {"embedding": [1, 0, 0]},
        {"expires_at": (datetime.now(UTC) - timedelta(days=1)).isoformat()},
        {"superseded_by": active["id"]},
    ):
        excluded.append(create_memory(retrieval_client, user_id, **fields))
    for embedding in (None, [0.0, 0.0]):
        legacy = create_memory(retrieval_client, user_id)
        with contextmanager(retrieval_client.app.dependency_overrides[get_db])() as db:
            db.get(Memory, UUID(legacy["id"])).embedding = embedding
            db.commit()
        excluded.append(legacy)
    response = retrieve(retrieval_client, user_id)
    assert response.status_code == 200, response.text
    assert [item["memory"]["id"] for item in response.json()["memories"]] == [active["id"]]
    for item in excluded:
        assert retrieval_client.get(f"/memories/{item['id']}").json()["access_count"] == 0


def test_semantic_candidate_pool_is_limited_to_twenty(retrieval_client):
    user_id = uuid4()
    for _ in range(20):
        create_memory(retrieval_client, user_id, importance=0, confidence=0)
    outside = create_memory(
        retrieval_client, user_id, importance=1, confidence=1, embedding=[0.9, 0.1]
    )
    results = retrieve(retrieval_client, user_id, limit=20).json()["memories"]
    assert len(results) == 20
    assert outside["id"] not in {item["memory"]["id"] for item in results}
    assert retrieval_client.get(f"/memories/{outside['id']}").json()["access_count"] == 0


def test_empty_results_and_custom_limit(retrieval_client):
    user_id = uuid4()
    assert retrieve(retrieval_client, user_id).json() == {"memories": []}
    create_memory(retrieval_client, user_id)
    create_memory(retrieval_client, user_id)
    assert len(retrieve(retrieval_client, user_id, limit=1).json()["memories"]) == 1


@pytest.mark.parametrize(
    "fields",
    [
        {"user_id": "bad"},
        {"query": " "},
        {"query": "x" * 8001},
        {"limit": 0},
        {"limit": 21},
        {"limit": True},
        {"limit": 1.5},
    ],
)
def test_invalid_requests_do_not_embed(retrieval_client, mock_llm, fields):
    response = retrieval_client.post(
        "/memory/retrieve", json={"user_id": str(uuid4()), "query": "Useful facts", **fields}
    )
    assert response.status_code == 422
    mock_llm.embed.assert_not_awaited()


@pytest.mark.parametrize(
    "embedding", [[], [0, 0], [float("nan"), 1], [float("inf"), 1], [1e39], [1e-100]]
)
def test_invalid_provider_vectors(retrieval_client, mock_llm, embedding):
    mock_llm.embed.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=embedding)]
    )
    assert retrieve(retrieval_client, uuid4()).status_code == 502


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
def test_embedding_failure_does_not_update_memories(retrieval_client, mock_llm, error, status):
    user_id = uuid4()
    memory = create_memory(retrieval_client, user_id)
    mock_llm.embed.side_effect = error
    response = retrieve(retrieval_client, user_id)
    assert response.status_code == status
    assert "private" not in response.text
    assert retrieval_client.get(f"/memories/{memory['id']}").json()["access_count"] == 0


def test_embedding_dimensions(mock_llm):
    service = EmbeddingService(mock_llm, "test-model", dimensions=2)
    assert asyncio.run(service.embed("hello")) == [1, 0]
    assert mock_llm.embed.call_args.kwargs["dimensions"] == 2
    with pytest.raises(EmbeddingError):
        asyncio.run(EmbeddingService(mock_llm, "test-model", dimensions=3).embed("hello"))


def test_postgres_cosine_sql():
    user_id = uuid4()
    statement = candidate_query(user_id, [1, 0], datetime.now(UTC))
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert "<=>" in sql
    assert "CASE WHEN" in sql
    assert "ORDER BY cosine_distance ASC, memories.id ASC" in sql
    assert "memories.user_id =" in sql
    assert 20 in compiled.params.values()
    assert user_id in compiled.params.values()


def test_update_failure_rolls_back(retrieval_client, monkeypatch):
    user_id = uuid4()
    memory = create_memory(retrieval_client, user_id)
    with monkeypatch.context() as patch:
        patch.setattr(
            Session, "commit", Mock(side_effect=OperationalError("commit", {}, Exception()))
        )
        response = retrieve(retrieval_client, user_id)
    assert response.status_code == 503
    stored = retrieval_client.get(f"/memories/{memory['id']}").json()
    assert stored["access_count"] == 0
    assert stored["last_accessed_at"] is None


def test_query_failure_rolls_back():
    db = Mock(spec=Session)
    db.execute.side_effect = OperationalError("select", {}, Exception())
    with pytest.raises(RetrievalError):
        retrieve_memories(db, uuid4(), [1, 0])
    db.rollback.assert_called_once()
    db.commit.assert_not_called()


def test_missing_configuration():
    with TestClient(create_app(Settings(_env_file=None))) as client:
        unavailable = client.app.state.llm_client
        client.app.state.llm_client = None
        assert retrieve(client, uuid4()).status_code == 503
        client.app.state.llm_client = unavailable
