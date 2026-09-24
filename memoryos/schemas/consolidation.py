from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StrictBool, StringConstraints

from memoryos.schemas.memory import MemoryResponse, Score


class ConsolidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: UUID
    after_id: UUID | None = None


class ConsolidationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    confidence: Score
    should_consolidate: StrictBool
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ConsolidatedMemory(BaseModel):
    memory: MemoryResponse
    source_memory_ids: list[UUID]


class ConsolidationResult(BaseModel):
    memories: list[ConsolidatedMemory]
    candidates_examined: int
    clusters_skipped: int
    next_cursor: UUID | None
