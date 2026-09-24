from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
)

from memoryos.models.memory import MemoryStatus, MemoryType

Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
VectorElement = Annotated[
    float, Field(ge=-3.4028234663852886e38, le=3.4028234663852886e38, allow_inf_nan=False)
]
Embedding = Annotated[list[VectorElement], Field(min_length=1, max_length=16000)]


class MemoryCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    user_id: UUID
    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    embedding: Embedding | None = None
    memory_type: MemoryType
    importance: Score = 0.5
    confidence: Score = 1.0
    expires_at: AwareDatetime | None = None
    source: (
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
        | None
    ) = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    superseded_by: UUID | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class MemoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: UUID
    content: str
    embedding: list[float] | None
    memory_type: MemoryType
    importance: float
    confidence: float
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime | None
    access_count: int
    expires_at: datetime | None
    source: str | None
    status: MemoryStatus
    superseded_by: UUID | None
    consolidated_at: datetime | None
    metadata: dict[str, JsonValue] = Field(validation_alias="metadata_")
