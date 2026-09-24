from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from memoryos.schemas.memory import MemoryResponse


class MemoryRetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: UUID
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
    limit: int = Field(default=5, ge=1, le=20, strict=True)
    debug: bool = False


class RetrievalComponents(BaseModel):
    semantic_similarity: float
    importance: float
    recency: float
    access_frequency: float
    confidence: float


class RetrievedMemory(BaseModel):
    memory: MemoryResponse
    score: float
    components: RetrievalComponents | None = None


class MemoryRetrievalResult(BaseModel):
    memories: list[RetrievedMemory]
