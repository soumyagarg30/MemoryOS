from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, StringConstraints


class MemoryRelationship(StrEnum):
    REINFORCEMENT = "REINFORCEMENT"
    CONTRADICTION = "CONTRADICTION"
    CONTEXT_SPECIFIC = "CONTEXT_SPECIFIC"
    UNRELATED = "UNRELATED"


class MemoryComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    memory_id: UUID
    relationship: MemoryRelationship
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class MemoryConflictAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", revalidate_instances="always")

    comparisons: list[MemoryComparison]
