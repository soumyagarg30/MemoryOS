from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from memoryos.api.extraction import get_llm_client
from memoryos.config import Settings
from memoryos.main import create_app
from memoryos.memory.prompts import EXTRACTION_PROMPT
from memoryos.schemas.extraction import MemoryCandidate, MemoryExtractionResult
from memoryos.services.ollama import APIConnectionError, APITimeoutError, RateLimitError


def candidate(**overrides):
    return {
        "content": "The user prefers concise answers.",
        "memory_type": "PREFERENCE",
        "importance": 0.7,
        "confidence": 0.95,
        "should_store": True,
        "reason": "An explicit preference that can improve future responses.",
        **overrides,
    }


def parsed_response(candidates):
    return SimpleNamespace(
        status="completed",
        output=[],
        output_parsed=MemoryExtractionResult(candidates=candidates),
    )


@pytest.fixture
def llm_client():
    client = Mock()
    client.parse = AsyncMock(return_value=parsed_response([candidate()]))
    return client


@pytest.fixture
def extraction_client(llm_client, monkeypatch):
    def forbid_database(*args, **kwargs):
        pytest.fail("Extraction must not connect to or write to the database")

    monkeypatch.setattr("psycopg.connect", forbid_database)
    monkeypatch.setattr("sqlalchemy.orm.Session.add", forbid_database)
    monkeypatch.setattr("sqlalchemy.orm.Session.commit", forbid_database)
    settings = Settings(
        _env_file=None, extraction_model="test-extraction-model"
    )
    app = create_app(settings)
    app.dependency_overrides[get_llm_client] = lambda: llm_client
    with TestClient(app) as client:
        yield client


def test_extract_candidates_without_persisting(extraction_client, llm_client):
    message = "Hello! I prefer concise answers."
    response = extraction_client.post("/memory/extract", json={"message": message})
    assert response.status_code == 200
    assert response.json() == {"candidates": [candidate()]}
    llm_client.parse.assert_awaited_once_with(
        model="test-extraction-model",
        input=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": message},
        ],
        text_format=MemoryExtractionResult,
        max_output_tokens=4096,
    )


@pytest.mark.parametrize("message", ["Hello!", "Thanks!", "Goodbye", "Okay"])
def test_trivial_messages_produce_no_memories(extraction_client, llm_client, message):
    llm_client.parse.return_value = parsed_response([])
    response = extraction_client.post("/memory/extract", json={"message": message})
    assert response.status_code == 200
    assert response.json() == {"candidates": []}
    prompt = llm_client.parse.call_args.kwargs["input"][0]["content"]
    assert "must never have should_store=true" in prompt
    assert "return an empty candidates list" in prompt


def test_rejected_candidates_are_returned_for_inspection(extraction_client, llm_client):
    rejected = candidate(should_store=False, confidence=0.1, reason="Insufficient evidence.")
    llm_client.parse.return_value = parsed_response([candidate(), rejected])
    response = extraction_client.post("/memory/extract", json={"message": "Some conversation"})
    assert response.status_code == 200
    assert response.json() == {"candidates": [candidate(), rejected]}


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"message": ""},
        {"message": " \n "},
        {"message": "x" * 20001},
        {"message": 12},
        {"message": "Hello", "user_id": "extra"},
    ],
)
def test_invalid_input_does_not_call_llm(extraction_client, llm_client, body):
    response = extraction_client.post("/memory/extract", json=body)
    assert response.status_code == 422
    llm_client.parse.assert_not_awaited()


@pytest.mark.parametrize(
    "fields",
    [
        {"importance": -0.1},
        {"importance": 1.1},
        {"confidence": -1},
        {"confidence": 2},
        {"importance": float("nan")},
        {"confidence": float("inf")},
        {"memory_type": "UNKNOWN"},
        {"content": " "},
        {"reason": ""},
        {"should_store": "true"},
        {"extra": "value"},
    ],
)
def test_invalid_candidate_is_rejected(extraction_client, llm_client, fields):
    llm_client.parse.return_value = SimpleNamespace(
        status="completed", output=[], output_parsed={"candidates": [candidate(**fields)]}
    )
    response = extraction_client.post("/memory/extract", json={"message": "Some fact"})
    assert response.status_code == 502
    assert response.json() == {"detail": "The model returned an invalid extraction result"}


@pytest.mark.parametrize("field", list(candidate()))
def test_candidate_fields_are_required(field):
    data = candidate()
    del data[field]
    with pytest.raises(ValidationError):
        MemoryCandidate.model_validate(data)


@pytest.mark.parametrize("memory_type", ["WORKING", "EPISODIC", "SEMANTIC", "PREFERENCE", "TASK"])
def test_all_candidate_types(memory_type):
    result = MemoryCandidate.model_validate(
        candidate(memory_type=memory_type, importance=0, confidence=1)
    )
    assert result.memory_type == memory_type


@pytest.mark.parametrize(
    "status, parsed",
    [
        ("completed", None),
        ("incomplete", None),
        ("incomplete", {"candidates": []}),
        ("completed", {"candidates": "not-a-list"}),
    ],
)
def test_missing_or_incomplete_output(extraction_client, llm_client, status, parsed):
    llm_client.parse.return_value = SimpleNamespace(
        status=status, output=[], output_parsed=parsed
    )
    assert (
        extraction_client.post("/memory/extract", json={"message": "Some fact"}).status_code == 502
    )


def test_refusal(extraction_client, llm_client):
    llm_client.parse.return_value = SimpleNamespace(
        status="completed",
        output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal")])],
        output_parsed=None,
    )
    response = extraction_client.post("/memory/extract", json={"message": "Some message"})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "error, expected_status",
    [
        (APITimeoutError(), 504),
        (APIConnectionError(), 502),
        (
            RateLimitError(
                "private provider detail",
            ),
            503,
        ),
        (ValueError("private malformed response"), 502),
    ],
)
def test_provider_errors_are_sanitized(extraction_client, llm_client, error, expected_status):
    llm_client.parse.side_effect = error
    response = extraction_client.post("/memory/extract", json={"message": "Some fact"})
    assert response.status_code == expected_status
    assert "private" not in response.text


def test_unconfigured_extraction():
    settings = Settings(_env_file=None)
    with TestClient(create_app(settings)) as client:
        unavailable = client.app.state.llm_client
        client.app.state.llm_client = None
        assert client.get("/health").status_code == 200
        response = client.post("/memory/extract", json={"message": "Some fact"})
        client.app.state.llm_client = unavailable
    assert response.status_code == 503
    assert response.json() == {"detail": "Memory extraction is not configured"}


def test_client_lifecycle(monkeypatch):
    client = Mock(close=AsyncMock())
    factory = Mock(return_value=client)
    monkeypatch.setattr("memoryos.main.OllamaClient", factory)
    settings = Settings(_env_file=None, ollama_base_url="http://localhost:11434")
    with TestClient(create_app(settings)):
        factory.assert_called_once_with(base_url="http://localhost:11434/", timeout=120.0)
        client.close.assert_not_awaited()
    client.close.assert_awaited_once()
