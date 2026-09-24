import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from memoryos.api.chat import get_chat_service
from memoryos.api.retrieval import get_embedding_service
from memoryos.schemas.chat import AssistantReply
from memoryos.schemas.conflicts import MemoryConflictAnalysis
from memoryos.schemas.extraction import MemoryExtractionResult
from memoryos.services.chat import ChatService
from memoryos.services.extraction import MemoryExtractionService
from memoryos.services.ollama import APIConnectionError, APITimeoutError


@pytest.fixture
def archive(memory_client, storage_llm, storage_service):
    calls = []

    async def parse(**kwargs):
        schema = kwargs["text_format"]
        calls.append(schema)
        if schema is AssistantReply:
            value = {"response": "Recorded your preference."}
        elif schema is MemoryExtractionResult:
            value = {
                "candidates": [
                    {
                        "content": "Uses PostgreSQL for new projects.",
                        "memory_type": "PREFERENCE",
                        "importance": 0.9,
                        "confidence": 0.95,
                        "should_store": True,
                        "reason": "Explicit preference",
                    }
                ]
            }
        else:
            context = json.loads(kwargs["input"][1]["content"])
            value = {
                "comparisons": [
                    {
                        "memory_id": m["id"],
                        "relationship": "REINFORCEMENT"
                        if "PostgreSQL" in m["content"]
                        else "CONTRADICTION",
                        "reason": "Explicit change",
                    }
                    for m in context["existing_memories"]
                ]
            }
        return SimpleNamespace(
            status="completed", output=[], output_parsed=schema.model_validate(value)
        )

    storage_llm.parse = AsyncMock(side_effect=parse)
    service = ChatService(
        storage_llm, storage_service, MemoryExtractionService(storage_llm, "test"), "test"
    )
    memory_client.app.dependency_overrides[get_chat_service] = lambda: service
    memory_client.app.dependency_overrides[get_embedding_service] = lambda: (
        storage_service.embeddings
    )
    return memory_client, storage_llm, calls


def test_chat_creates_reinforces_and_orders_processing(archive):
    client, llm, calls = archive
    user = str(uuid4())
    payload = {"user_id": user, "message": "I use PostgreSQL."}
    first = client.post("/chat", json=payload)
    assert first.status_code == 200
    assert len(first.json()["created_memories"]) == 1
    assert calls[:2] == [AssistantReply, MemoryExtractionResult]
    second = client.post("/chat", json=payload).json()
    assert second["created_memories"] == []
    assert len(second["reinforced_memories"]) == 1
    assert second["retrieved_memories"][0]["components"] is not None
    assert "MEMORY_REINFORCED" in [e["kind"] for e in second["memory_activity_events"]]
    assert calls[-1] is MemoryConflictAnalysis


def test_conflict_events_and_provenance(archive):
    client, _, _ = archive
    user = str(uuid4())
    old = client.post(
        "/memories", json={"user_id": user, "content": "Uses MongoDB", "memory_type": "PREFERENCE"}
    ).json()
    response = client.post("/chat", json={"user_id": user, "message": "Switched to PostgreSQL"})
    assert response.status_code == 200
    result = response.json()
    assert result["superseded_memories"][0]["id"] == old["id"]
    assert result["conflict_actions"][0]["relationship"] == "CONTRADICTION"
    kinds = [e["kind"] for e in result["memory_activity_events"]]
    assert "CONFLICT_DETECTED" in kinds and "MEMORY_SUPERSEDED" in kinds
    new = result["created_memories"][0]
    provenance = client.get(f"/memories/{new['id']}/provenance").json()
    assert provenance["supersedes"][0]["id"] == old["id"]


@pytest.mark.parametrize(
    "error,status", [(APIConnectionError("secret"), 502), (APITimeoutError("secret"), 504)]
)
def test_chat_provider_failure(archive, error, status):
    client, llm, _ = archive
    llm.parse.side_effect = error
    response = client.post("/chat", json={"user_id": str(uuid4()), "message": "hi"})
    assert response.status_code == status
    assert "secret" not in response.text
    assert client.get("/memories").json() == []


def test_extraction_failure_preserves_reply(archive):
    client, llm, _ = archive
    llm.parse.side_effect = [
        SimpleNamespace(status="completed", output_parsed=AssistantReply(response="Hello")),
        APIConnectionError(),
    ]
    response = client.post("/chat", json={"user_id": str(uuid4()), "message": "hi"}).json()
    assert response["assistant_response"] == "Hello"
    assert response["warnings"]
    assert response["created_memories"] == []


def test_empty_extraction(archive):
    client, llm, _ = archive
    llm.parse.side_effect = [
        SimpleNamespace(status="completed", output_parsed=AssistantReply(response="Hello")),
        SimpleNamespace(
            status="completed", output=[], output_parsed=MemoryExtractionResult(candidates=[])
        ),
    ]
    response = client.post("/chat", json={"user_id": str(uuid4()), "message": "hi"}).json()
    assert response["warnings"] == []
    assert response["memory_activity_events"][-1]["kind"] == "NO_MEMORY"


def test_empty_archive_does_not_report_recall_failure(archive):
    client, _, _ = archive
    response = client.post(
        "/chat",
        json={"user_id": str(uuid4()), "message": "Remember that I use Python at work."},
    ).json()
    assert response["retrieved_memories"] == []
    assert "Recall unavailable; this response has no recalled context." not in response["warnings"]


def test_chat_preserves_episode_context_for_recall(archive):
    client, llm, _ = archive
    user = str(uuid4())
    episode = client.post(
        "/memories",
        json={
            "user_id": user,
            "content": "Visited Paris for a conference.",
            "memory_type": "EPISODIC",
            "metadata": {"event_date": "2025-06-12", "context": "work"},
            "source": "journal",
            "confidence": 0.85,
        },
    ).json()
    response = client.post(
        "/chat", json={"user_id": user, "message": "When and why did I visit Paris?"}
    )
    assert response.status_code == 200
    reply_call = next(
        call.kwargs for call in llm.parse.await_args_list
        if call.kwargs["text_format"] is AssistantReply
    )
    recalled = json.loads(reply_call["input"][1]["content"])["memories"][0]
    assert recalled["id"] == episode["id"]
    assert recalled["type"] == "EPISODIC"
    assert recalled["metadata"] == episode["metadata"]
    assert recalled["created_at"].replace("+00:00", "Z") == episode["created_at"]
    assert recalled["source"] == "journal"
    assert recalled["confidence"] == 0.85


def test_demo_repeat_and_user_isolation(archive):
    client, _, _ = archive
    user = str(uuid4())
    first = client.post("/demo", json={"user_id": user})
    assert first.status_code == 200
    assert len(first.json()["memories"]) == 6
    second = client.post("/demo", json={"user_id": user}).json()
    assert second["already_loaded"] is True
    assert second["memory_activity_events"] == []
    assert len(client.get("/memories", params={"user_id": user}).json()) == 6
    assert client.get("/memories", params={"user_id": str(uuid4())}).json() == []


def test_decay_and_invalid_ids(archive):
    client, _, _ = archive
    assert client.get("/memories/bad/provenance").status_code == 422
    assert client.get(f"/memories/{uuid4()}/provenance").status_code == 404
    assert client.post("/memory/decay", json={"user_id": str(uuid4())}).json()["processed"] == 0
    assert client.post("/chat", json={"user_id": "bad", "message": ""}).status_code == 422
