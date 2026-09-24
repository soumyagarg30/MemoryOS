from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel


class APIError(Exception):
    pass


class APIConnectionError(APIError):
    pass


class APITimeoutError(APIError):
    pass


class RateLimitError(APIError):
    pass


class BadRequestError(APIError):
    pass


@dataclass
class ParsedResult:
    output_parsed: BaseModel
    status: str = "completed"
    output: list[Any] = field(default_factory=list)


@dataclass
class EmbeddingItem:
    embedding: list[float]


@dataclass
class EmbeddingResult:
    data: list[EmbeddingItem]


class OllamaClient:
    """Native Ollama transport with validated, non-streaming structured output."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            transport=transport,
            trust_env=False,
        )

    async def close(self) -> None:
        await self.http.aclose()

    async def _post(self, path: str, payload: dict) -> dict:
        try:
            response = await self.http.post(path, json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise APITimeoutError("Ollama request timed out") from exc
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            error = (
                RateLimitError
                if code in (429, 503)
                else (BadRequestError if code in (400, 422) else APIError)
            )
            raise error("Ollama request failed") from exc
        except httpx.RequestError as exc:
            raise APIConnectionError("Ollama is unavailable") from exc
        data = response.json()
        if not isinstance(data, dict) or data.get("error"):
            raise ValueError("Invalid Ollama response")
        return data

    async def parse(
        self,
        *,
        model: str,
        input: list[dict[str, str]],
        text_format: type[BaseModel],
        max_output_tokens: int,
    ) -> ParsedResult:
        data = await self._post(
            "api/chat",
            {
                "model": model,
                "messages": input,
                "stream": False,
                "format": text_format.model_json_schema(),
                "options": {"temperature": 0, "num_predict": max_output_tokens},
            },
        )
        if data.get("done") is not True or data.get("done_reason") != "stop":
            raise ValueError("Incomplete Ollama output")
        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("Missing Ollama output")
        return ParsedResult(text_format.model_validate_json(message["content"]))

    async def embed(
        self,
        *,
        model: str,
        input: str,
        dimensions: int | None = None,
    ) -> EmbeddingResult:
        payload = {"model": model, "input": input, "truncate": False}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        data = await self._post("api/embed", payload)
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or not all(isinstance(v, list) for v in vectors):
            raise ValueError("Invalid Ollama embeddings")
        return EmbeddingResult([EmbeddingItem(v) for v in vectors])
