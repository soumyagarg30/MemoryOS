from pydantic import ValidationError

from memoryos.memory.prompts import EXTRACTION_PROMPT
from memoryos.schemas.extraction import MemoryExtractionResult
from memoryos.services.ollama import APIError, APITimeoutError, OllamaClient, RateLimitError


class ExtractionError(Exception):
    pass


class ExtractionTimeout(ExtractionError):
    pass


class ExtractionBusy(ExtractionError):
    pass


class ExtractionRefused(ExtractionError):
    pass


class InvalidExtractionOutput(ExtractionError):
    pass


class MemoryExtractionService:
    def __init__(self, client: OllamaClient, model: str) -> None:
        self.client = client
        self.model = model

    async def extract(self, message: str) -> MemoryExtractionResult:
        try:
            response = await self.client.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": message},
                ],
                text_format=MemoryExtractionResult,
                max_output_tokens=4096,
            )
        except APITimeoutError as exc:
            raise ExtractionTimeout from exc
        except RateLimitError as exc:
            raise ExtractionBusy from exc
        except APIError as exc:
            raise ExtractionError from exc
        except (ValidationError, ValueError) as exc:
            raise InvalidExtractionOutput from exc

        if response.status != "completed":
            raise InvalidExtractionOutput
        for item in response.output:
            if item.type == "message" and any(part.type == "refusal" for part in item.content):
                raise ExtractionRefused
        if response.output_parsed is None:
            raise InvalidExtractionOutput
        try:
            return MemoryExtractionResult.model_validate(response.output_parsed)
        except ValidationError as exc:
            raise InvalidExtractionOutput from exc
