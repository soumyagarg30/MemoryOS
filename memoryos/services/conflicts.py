import json

from pydantic import ValidationError

from memoryos.memory.prompts import CONFLICT_PROMPT
from memoryos.models import Memory
from memoryos.schemas.conflicts import MemoryConflictAnalysis
from memoryos.schemas.memory import MemoryCreate
from memoryos.services.ollama import APIError, APITimeoutError, OllamaClient, RateLimitError


class ConflictDetectionError(Exception):
    pass


class ConflictDetectionTimeout(ConflictDetectionError):
    pass


class ConflictDetectionBusy(ConflictDetectionError):
    pass


class ConflictDetectionRefused(ConflictDetectionError):
    pass


class InvalidConflictAnalysis(ConflictDetectionError):
    pass


class MemoryConflictDetector:
    def __init__(self, client: OllamaClient, model: str) -> None:
        self.client = client
        self.model = model

    async def classify(
        self, proposed: MemoryCreate, existing: list[Memory]
    ) -> MemoryConflictAnalysis:
        if not existing:
            return MemoryConflictAnalysis(comparisons=[])
        context = {
            "new_memory": proposed.model_dump(mode="json", exclude={"embedding", "user_id"}),
            "existing_memories": [
                {
                    "id": str(memory.id),
                    "content": memory.content,
                    "memory_type": memory.memory_type.value,
                    "source": memory.source,
                    "metadata": memory.metadata_,
                    "created_at": memory.created_at.isoformat(),
                    "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
                }
                for memory in existing
            ],
        }
        try:
            response = await self.client.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": CONFLICT_PROMPT},
                    {"role": "user", "content": json.dumps(context, allow_nan=False)},
                ],
                text_format=MemoryConflictAnalysis,
                max_output_tokens=4096,
            )
        except APITimeoutError as exc:
            raise ConflictDetectionTimeout from exc
        except RateLimitError as exc:
            raise ConflictDetectionBusy from exc
        except APIError as exc:
            raise ConflictDetectionError from exc
        except (ValidationError, ValueError) as exc:
            raise InvalidConflictAnalysis from exc

        if response.status != "completed":
            raise InvalidConflictAnalysis
        if any(
            item.type == "message" and any(part.type == "refusal" for part in item.content)
            for item in response.output
        ):
            raise ConflictDetectionRefused
        try:
            result = MemoryConflictAnalysis.model_validate(response.output_parsed)
        except ValidationError as exc:
            raise InvalidConflictAnalysis from exc

        ids = [comparison.memory_id for comparison in result.comparisons]
        if len(ids) != len(set(ids)) or set(ids) != {memory.id for memory in existing}:
            raise InvalidConflictAnalysis
        return result
