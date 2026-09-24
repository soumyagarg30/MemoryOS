import math

from pgvector import Vector
from pydantic import TypeAdapter, ValidationError

from memoryos.schemas.memory import Embedding
from memoryos.services.ollama import (
    APIError,
    APITimeoutError,
    BadRequestError,
    OllamaClient,
    RateLimitError,
)


class EmbeddingError(Exception):
    pass


class EmbeddingTimeout(EmbeddingError):
    pass


class EmbeddingBusy(EmbeddingError):
    pass


class EmbeddingRejected(EmbeddingError):
    pass


class EmbeddingService:
    def __init__(self, client: OllamaClient, model: str, dimensions: int | None = None) -> None:
        self.client = client
        self.model = model
        self.dimensions = dimensions

    async def embed(self, query: str) -> list[float]:
        options = {"dimensions": self.dimensions} if self.dimensions is not None else {}
        try:
            response = await self.client.embed(
                model=self.model, input=query, **options
            )
        except APITimeoutError as exc:
            raise EmbeddingTimeout from exc
        except RateLimitError as exc:
            raise EmbeddingBusy from exc
        except BadRequestError as exc:
            raise EmbeddingRejected from exc
        except APIError as exc:
            raise EmbeddingError from exc
        except (ValidationError, ValueError) as exc:
            raise EmbeddingError from exc

        if len(response.data) != 1:
            raise EmbeddingError
        try:
            vector = Vector(
                TypeAdapter(Embedding).validate_python(response.data[0].embedding)
            ).to_list()
        except ValidationError as exc:
            raise EmbeddingError from exc
        if not math.hypot(*vector) or (
            self.dimensions is not None and len(vector) != self.dimensions
        ):
            raise EmbeddingError
        return vector
