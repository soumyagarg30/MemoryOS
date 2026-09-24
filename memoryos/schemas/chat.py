from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from memoryos.schemas.conflicts import MemoryComparison
from memoryos.schemas.memory import MemoryResponse
from memoryos.schemas.retrieval import RetrievedMemory


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: UUID
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]


class AssistantReply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    response: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12000)
    ]


class ActivityEvent(BaseModel):
    kind: str
    message: str
    memory_ids: list[UUID] = Field(default_factory=list)
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatResult(BaseModel):
    assistant_response: str
    retrieved_memories: list[RetrievedMemory] = Field(default_factory=list)
    created_memories: list[MemoryResponse] = Field(default_factory=list)
    reinforced_memories: list[MemoryResponse] = Field(default_factory=list)
    superseded_memories: list[MemoryResponse] = Field(default_factory=list)
    conflict_actions: list[MemoryComparison] = Field(default_factory=list)
    memory_activity_events: list[ActivityEvent] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
