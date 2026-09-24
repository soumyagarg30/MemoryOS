from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints

from memoryos.models.memory import MemoryType
from memoryos.schemas.memory import Score

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class MemoryExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: Annotated[NonEmptyText, Field(max_length=20000)]


class MemoryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    content: NonEmptyText
    memory_type: MemoryType
    importance: Score
    confidence: Score
    should_store: StrictBool
    reason: NonEmptyText


class MemoryExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    candidates: list[MemoryCandidate]
