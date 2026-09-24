import json
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from memoryos.config import Settings
from memoryos.db.session import get_db
from memoryos.main import create_app
from memoryos.memory.prompts import CONFLICT_PROMPT
from memoryos.models import Memory
from memoryos.schemas.conflicts import MemoryConflictAnalysis
from memoryos.schemas.memory import MemoryCreate
from memoryos.services.ollama import APIConnectionError, APITimeoutError, RateLimitError
from memoryos.services.storage import MemoryWriteBusy, prepare_write


def post(client, user_id, content="I prefer tea.", **fields):
    return client.post(
        "/memories",
        json={
            "user_id": str(user_id),
            "content": content,
            "memory_type": "PREFERENCE",
            "confidence": 0.6,
            "embedding": [1.0, 0.0],
            **fields,
        },
    )


def seed(client, user_id, **fields):
    response = post(client, user_id, **fields)
    assert response.status_code == 201, response.text
    return response.json()


def stored(client, memory):
    return client.get(f"/memories/{memory['id']}").json()


def comparisons_response(comparisons):
    return SimpleNamespace(
        status="completed", output=[], output_parsed={"comparisons": comparisons}
    )


def classify_as(llm, relationship):
    async def parse(**kwargs):
        context = json.loads(kwargs["input"][1]["content"])
        return comparisons_response(
            [
                {
                    "memory_id": memory["id"],
                    "relationship": relationship.get(memory["id"], "UNRELATED")
                    if isinstance(relationship, dict)
                    else relationship,
                    "reason": "Mocked semantic classification.",
                }
                for memory in context["existing_memories"]
            ]
        )

    llm.parse.side_effect = parse


def test_contradiction_supersedes_old_memory(memory_client, storage_llm):
    user_id = uuid4()
    old = seed(memory_client, user_id, content="I live in Paris.")
    classify_as(storage_llm, "CONTRADICTION")
    response = post(memory_client, user_id, "I now live in Berlin.")
    assert response.status_code == 201
    new = response.json()
    assert new["id"] != old["id"]
    assert new["status"] == "ACTIVE"
    assert new["superseded_by"] is None
    previous = stored(memory_client, old)
    assert previous["status"] == "SUPERSEDED"
    assert previous["superseded_by"] == new["id"]
    assert previous["content"] == old["content"]
    assert previous["confidence"] == old["confidence"]
    assert previous["access_count"] == 0
    active = memory_client.get(
        "/memories", params={"user_id": str(user_id), "status": "ACTIVE"}
    ).json()
    assert [item["id"] for item in active] == [new["id"]]


@pytest.mark.parametrize("confidence, expected", [(0, 0.05), (0.6, 0.65), (0.98, 1), (1, 1)])
def test_reinforcement_reuses_record_and_caps_confidence(
    memory_client, storage_llm, confidence, expected
):
    user_id = uuid4()
    old = seed(memory_client, user_id, confidence=confidence, metadata={"context": "home"})
    classify_as(storage_llm, "REINFORCEMENT")
    response = post(memory_client, user_id, "Tea is my preference.")
    assert response.status_code == 200
    result = response.json()
    assert result["id"] == old["id"]
    assert result["confidence"] == pytest.approx(expected)
    assert response.headers["location"] == f"/memories/{old['id']}"
    for field in (
        "content",
        "metadata",
        "importance",
        "created_at",
        "embedding",
        "access_count",
        "last_accessed_at",
    ):
        assert result[field] == old[field]
    assert len(memory_client.get("/memories").json()) == 1
    assert stored(memory_client, old)["confidence"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "old_content, new_content, relationship",
    [
        ("I drink coffee at work.", "I prefer tea at home.", "CONTEXT_SPECIFIC"),
        ("I lived in Paris in 2020.", "I lived in Berlin in 2024.", "CONTEXT_SPECIFIC"),
        ("I prefer tea.", "I prefer green tea without sugar.", "CONTEXT_SPECIFIC"),
        ("I prefer tea.", "My next task is to book a flight.", "UNRELATED"),
    ],
)
def test_compatible_and_unrelated_memories_preserve_both(
    memory_client, storage_llm, old_content, new_content, relationship
):
    user_id = uuid4()
    old = seed(memory_client, user_id, content=old_content)
    classify_as(storage_llm, relationship)
    response = post(memory_client, user_id, new_content)
    assert response.status_code == 201
    assert response.json()["id"] != old["id"]
    assert stored(memory_client, old) == old
    assert len(memory_client.get("/memories").json()) == 2


def test_multiple_contradictions_share_one_replacement(memory_client, storage_llm):
    user_id = uuid4()
    old = [seed(memory_client, user_id, content=f"Old fact {i}") for i in range(3)]
    classify_as(storage_llm, "CONTRADICTION")
    new = post(memory_client, user_id, "Updated fact").json()
    for memory in old:
        assert stored(memory_client, memory)["superseded_by"] == new["id"]
        assert stored(memory_client, memory)["status"] == "SUPERSEDED"
    assert len(memory_client.get("/memories").json()) == 4


def test_multiple_reinforcements_do_not_create_another_duplicate(memory_client, storage_llm):
    user_id = uuid4()
    nearest = seed(memory_client, user_id)
    other = seed(memory_client, user_id, embedding=[0.9, 0.1])
    classify_as(storage_llm, "REINFORCEMENT")
    response = post(memory_client, user_id)
    assert response.status_code == 200
    assert response.json()["id"] == nearest["id"]
    for memory in (nearest, other):
        assert stored(memory_client, memory)["confidence"] == pytest.approx(0.65)
    assert len(memory_client.get("/memories").json()) == 2


def test_mixed_relationships_reuse_canonical_memory(memory_client, storage_llm):
    user_id = uuid4()
    reinforced = seed(memory_client, user_id)
    contradicted = seed(memory_client, user_id, content="I dislike tea.")
    compatible = seed(memory_client, user_id, content="At work I drink coffee.")
    unrelated = seed(memory_client, user_id, content="Book a flight.")
    classify_as(
        storage_llm,
        {
            reinforced["id"]: "REINFORCEMENT",
            contradicted["id"]: "CONTRADICTION",
            compatible["id"]: "CONTEXT_SPECIFIC",
            unrelated["id"]: "UNRELATED",
        },
    )
    response = post(memory_client, user_id)
    assert response.status_code == 200
    assert response.json()["id"] == reinforced["id"]
    assert response.json()["confidence"] == pytest.approx(0.65)
    assert stored(memory_client, contradicted)["superseded_by"] == reinforced["id"]
    assert stored(memory_client, compatible) == compatible
    assert stored(memory_client, unrelated) == unrelated
    assert len(memory_client.get("/memories").json()) == 4


def test_structured_request_includes_context_and_exact_ids(memory_client, storage_llm):
    user_id = uuid4()
    old = seed(memory_client, user_id, source="user", metadata={"place": "work"})
    post(memory_client, user_id, "I prefer tea at home.", metadata={"place": "home"})
    call = storage_llm.parse.call_args.kwargs
    assert call["model"] == "test-conflict-model"
    assert call["text_format"] is MemoryConflictAnalysis
    assert call["input"][0] == {"role": "system", "content": CONFLICT_PROMPT}
    data = json.loads(call["input"][1]["content"])
    assert data["existing_memories"][0]["id"] == old["id"]
    assert data["existing_memories"][0]["metadata"] == {"place": "work"}
    assert data["new_memory"]["metadata"] == {"place": "home"}
    assert "embedding" not in data["new_memory"]


def test_no_candidates_skips_llm_and_generates_missing_embedding(memory_client, storage_llm):
    response = post(memory_client, uuid4(), embedding=None)
    assert response.status_code == 201
    assert response.json()["embedding"] == [1, 0]
    storage_llm.embed.assert_awaited_once()
    storage_llm.parse.assert_not_awaited()


def test_supplied_embedding_does_not_call_embedder(memory_client, storage_llm):
    assert post(memory_client, uuid4()).status_code == 201
    storage_llm.embed.assert_not_awaited()


@pytest.mark.parametrize("vector", [[0, 0], [1e-100]])
def test_zero_vector_is_rejected(memory_client, storage_llm, vector):
    assert post(memory_client, uuid4(), embedding=vector).status_code == 422
    assert memory_client.get("/memories").json() == []
    storage_llm.parse.assert_not_awaited()


@pytest.mark.parametrize(
    "fields",
    [
        {"status": "ARCHIVED"},
        {"status": "STALE"},
        {"status": "SUPERSEDED"},
        {"expires_at": "2000-01-01T00:00:00Z"},
    ],
)
def test_historical_incoming_record_does_not_replace_current_memory(
    memory_client, storage_llm, fields
):
    user_id = uuid4()
    old = seed(memory_client, user_id)
    classify_as(storage_llm, "CONTRADICTION")
    response = post(memory_client, user_id, **fields)
    assert response.status_code == 201
    assert stored(memory_client, old) == old
    storage_llm.parse.assert_not_awaited()


def test_candidate_filtering_and_user_isolation(memory_client, storage_llm):
    user_id = uuid4()
    excluded = [seed(memory_client, uuid4())]
    for fields in (
        {"status": "ARCHIVED"},
        {"status": "STALE"},
        {"status": "SUPERSEDED"},
        {"expires_at": "2000-01-01T00:00:00Z"},
        {"embedding": [1, 0, 0]},
    ):
        excluded.append(seed(memory_client, user_id, **fields))
    legacy = seed(memory_client, user_id)
    with contextmanager(memory_client.app.dependency_overrides[get_db])() as db:
        db.get(Memory, UUID(legacy["id"])).embedding = None
        db.commit()
    excluded.append(stored(memory_client, legacy))
    eligible = seed(memory_client, user_id)
    excluded.append(seed(memory_client, user_id, superseded_by=eligible["id"]))
    storage_llm.parse.reset_mock()
    classify_as(storage_llm, "CONTRADICTION")
    response = post(memory_client, user_id)
    assert response.status_code == 201
    data = json.loads(storage_llm.parse.call_args.kwargs["input"][1]["content"])
    assert [item["id"] for item in data["existing_memories"]] == [eligible["id"]]
    for memory in excluded:
        assert stored(memory_client, memory) == memory


def test_only_twenty_semantic_candidates_are_classified(memory_client, storage_llm):
    user_id = uuid4()
    for _ in range(20):
        seed(memory_client, user_id)
    outside = seed(memory_client, user_id, embedding=[0, 1])
    classify_as(storage_llm, "CONTRADICTION")
    assert post(memory_client, user_id).status_code == 201
    data = json.loads(storage_llm.parse.call_args.kwargs["input"][1]["content"])
    assert len(data["existing_memories"]) == 20
    assert outside["id"] not in {item["id"] for item in data["existing_memories"]}
    assert stored(memory_client, outside) == outside


@pytest.mark.parametrize(
    "malformation",
    [
        "missing",
        "duplicate",
        "unknown",
        "extra",
        "bad_relation",
        "empty_reason",
        "missing_reason",
        "none",
    ],
)
def test_malformed_analysis_never_writes(memory_client, storage_llm, malformation):
    user_id = uuid4()
    old = seed(memory_client, user_id)
    comparison = {"memory_id": old["id"], "relationship": "CONTRADICTION", "reason": "Conflict."}
    comparisons = [comparison]
    if malformation == "missing":
        comparisons = []
    elif malformation == "duplicate":
        comparisons = [comparison, comparison]
    elif malformation == "unknown":
        comparison["memory_id"] = str(uuid4())
    elif malformation == "extra":
        comparisons.append({**comparison, "memory_id": str(uuid4())})
    elif malformation == "bad_relation":
        comparison["relationship"] = "DELETE"
    elif malformation == "empty_reason":
        comparison["reason"] = " "
    elif malformation == "missing_reason":
        del comparison["reason"]
    response = comparisons_response(comparisons)
    if malformation == "none":
        response.output_parsed = None
    storage_llm.parse.side_effect = None
    storage_llm.parse.return_value = response
    result = post(memory_client, user_id)
    assert result.status_code == 502
    assert stored(memory_client, old) == old
    assert len(memory_client.get("/memories").json()) == 1


@pytest.mark.parametrize("kind, status", [("incomplete", 502), ("refusal", 422)])
def test_incomplete_or_refused_analysis(memory_client, storage_llm, kind, status):
    user_id = uuid4()
    old = seed(memory_client, user_id)
    response = SimpleNamespace(
        status="incomplete" if kind == "incomplete" else "completed", output=[], output_parsed=None
    )
    if kind == "refusal":
        response.output = [
            SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal")])
        ]
    storage_llm.parse.side_effect = None
    storage_llm.parse.return_value = response
    assert post(memory_client, user_id).status_code == status
    assert stored(memory_client, old) == old
    assert len(memory_client.get("/memories").json()) == 1


@pytest.mark.parametrize(
    "error, status",
    [
        (APITimeoutError(), 504),
        (APIConnectionError(), 502),
        (
            RateLimitError(
                "private provider detail",
            ),
            503,
        ),
        (ValueError("private invalid JSON"), 502),
    ],
)
def test_provider_errors_leave_database_unchanged(memory_client, storage_llm, error, status):
    user_id = uuid4()
    old = seed(memory_client, user_id)
    storage_llm.parse.side_effect = error
    response = post(memory_client, user_id)
    assert response.status_code == status
    assert "private" not in response.text
    assert stored(memory_client, old) == old
    assert len(memory_client.get("/memories").json()) == 1


@pytest.mark.parametrize("relationship", ["CONTRADICTION", "REINFORCEMENT", "CONTEXT_SPECIFIC"])
@pytest.mark.parametrize(
    "failure, status",
    [
        (OperationalError("commit", {}, Exception()), 503),
        (IntegrityError("commit", {}, Exception()), 409),
    ],
)
def test_commit_failure_rolls_back_all_changes(
    memory_client, storage_llm, monkeypatch, relationship, failure, status
):
    user_id = uuid4()
    old = seed(memory_client, user_id)
    classify_as(storage_llm, relationship)
    with monkeypatch.context() as patch:
        patch.setattr(Session, "commit", Mock(side_effect=failure))
        response = post(memory_client, user_id)
    assert response.status_code == status
    assert stored(memory_client, old) == old
    assert len(memory_client.get("/memories").json()) == 1


def test_embedding_failure_does_not_insert(memory_client, storage_llm):
    storage_llm.embed.side_effect = APITimeoutError(
    )
    assert post(memory_client, uuid4(), embedding=None).status_code == 504
    assert memory_client.get("/memories").json() == []
    storage_llm.parse.assert_not_awaited()


def test_expiration_during_analysis_aborts_write(memory_client, storage_llm, monkeypatch):
    user_id = uuid4()
    now = datetime.now(UTC)
    old = seed(memory_client, user_id, expires_at=(now + timedelta(minutes=1)).isoformat())
    classify_as(storage_llm, "CONTRADICTION")
    clock = Mock(wraps=datetime)
    clock.now.side_effect = [now, now + timedelta(minutes=2)]
    monkeypatch.setattr("memoryos.services.storage.datetime", clock)
    assert post(memory_client, user_id).status_code == 409
    assert stored(memory_client, old) == old
    assert len(memory_client.get("/memories").json()) == 1


def test_postgres_user_lock_rejects_concurrent_writer():
    db = Mock(spec=Session)
    db.get_bind.return_value.dialect.name = "postgresql"
    db.scalar.return_value = False
    payload = MemoryCreate(
        user_id=uuid4(), content="A fact", memory_type="SEMANTIC", embedding=[1, 0]
    )
    with pytest.raises(MemoryWriteBusy):
        prepare_write(db, payload)
    db.execute.assert_not_called()
    statement = db.scalar.call_args.args[0].compile(dialect=postgresql.dialect())
    assert "pg_try_advisory_xact_lock" in str(statement)
    assert -(2**63) <= next(iter(statement.params.values())) < 2**63


def test_postgres_candidate_rows_are_locked():
    db = Mock(spec=Session)
    db.get_bind.return_value.dialect.name = "postgresql"
    db.scalar.return_value = True
    db.execute.return_value = []
    payload = MemoryCreate(
        user_id=uuid4(), content="A fact", memory_type="SEMANTIC", embedding=[1, 0]
    )
    assert prepare_write(db, payload) == []
    statement = db.execute.call_args.args[0].compile(dialect=postgresql.dialect())
    assert "FOR UPDATE" in str(statement)
    assert "<=>" in str(statement)
    assert payload.user_id in statement.params.values()


def test_storage_without_client_fails_closed():
    with TestClient(create_app(Settings(_env_file=None))) as client:
        unavailable = client.app.state.llm_client
        client.app.state.llm_client = None
        assert post(client, uuid4()).status_code == 503
        assert client.get("/health").status_code == 200
        client.app.state.llm_client = unavailable
