import asyncio
import json

import httpx
import pytest

from memoryos.schemas.extraction import MemoryExtractionResult
from memoryos.services.embeddings import EmbeddingService
from memoryos.services.extraction import MemoryExtractionService
from memoryos.services.ollama import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    OllamaClient,
    RateLimitError,
)


def run_with_transport(handler, operation):
    async def run():
        client = OllamaClient("http://ollama.test", transport=httpx.MockTransport(handler))
        try:
            return await operation(client)
        finally:
            await client.close()

    return asyncio.run(run())


def extract(client):
    return MemoryExtractionService(client, "qwen2.5:7b").extract("Hello!")


def test_native_structured_output():
    def handler(request):
        assert str(request.url) == "http://ollama.test/api/chat"
        assert "authorization" not in request.headers
        body = json.loads(request.content)
        assert body["model"] == "qwen2.5:7b"
        assert body["format"] == MemoryExtractionResult.model_json_schema()
        assert body["messages"][-1] == {"role": "user", "content": "Hello!"}
        assert body["stream"] is False
        assert body["options"] == {"temperature": 0, "num_predict": 4096}
        return httpx.Response(
            200,
            json={
                "done": True,
                "done_reason": "stop",
                "message": {"content": '{"candidates": []}'},
            },
        )

    assert run_with_transport(handler, extract).candidates == []


@pytest.mark.parametrize(
    "body",
    [
        {},
        [],
        {"error": "private"},
        {"done": False, "done_reason": "stop"},
        {"done": True, "done_reason": "length", "message": {"content": '{"candidates": []}'}},
        {"done": True, "done_reason": "stop", "message": {}},
        {"done": True, "done_reason": "stop", "message": {"content": "not JSON"}},
        {"done": True, "done_reason": "stop", "message": {"content": '{"candidates": [1]}'}},
    ],
)
def test_invalid_chat_output(body):
    from memoryos.services.extraction import InvalidExtractionOutput

    with pytest.raises(InvalidExtractionOutput):
        run_with_transport(lambda _: httpx.Response(200, json=body), extract)


@pytest.mark.parametrize(
    "status,error",
    [
        (400, BadRequestError),
        (422, BadRequestError),
        (429, RateLimitError),
        (503, RateLimitError),
        (404, APIError),
        (500, APIError),
    ],
)
def test_http_errors(status, error):
    with pytest.raises(error):
        run_with_transport(
            lambda _: httpx.Response(status, json={"error": "private details"}),
            lambda client: client.embed(model="embed", input="fact"),
        )


@pytest.mark.parametrize(
    "cause,error",
    [
        (httpx.ReadTimeout, APITimeoutError),
        (httpx.ConnectError, APIConnectionError),
    ],
)
def test_connection_errors(cause, error):
    def handler(request):
        raise cause("private details", request=request)

    with pytest.raises(error):
        run_with_transport(handler, lambda client: client.embed(model="embed", input="fact"))


def test_native_embedding_request():
    def handler(request):
        assert request.url.path == "/api/embed"
        assert json.loads(request.content) == {
            "model": "nomic-embed-text:latest",
            "input": "fact",
            "truncate": False,
            "dimensions": 2,
        }
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    result = run_with_transport(
        handler,
        lambda client: EmbeddingService(
            client,
            "nomic-embed-text:latest",
            dimensions=2,
        ).embed("fact"),
    )
    assert result == [1.0, 0.0]


@pytest.mark.parametrize("body", [{}, {"embeddings": None}, {"embeddings": [1]}])
def test_malformed_embedding_response(body):
    with pytest.raises(ValueError):
        run_with_transport(
            lambda _: httpx.Response(200, json=body),
            lambda client: client.embed(model="embed", input="fact"),
        )
