from uuid import UUID

from pydantic import BaseModel, ConfigDict

from memoryos.schemas.chat import ActivityEvent
from memoryos.schemas.memory import MemoryResponse


class ArchiveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: UUID


class DecayRequest(ArchiveRequest):
    dry_run: bool = False


class ProvenanceResult(BaseModel):
    memory_id: UUID
    derived_from: list[MemoryResponse]
    consolidated_into: list[MemoryResponse]
    supersedes: list[MemoryResponse]
    superseded_by: MemoryResponse | None


class DemoResult(BaseModel):
    memories: list[MemoryResponse]
    already_loaded: bool
    memory_activity_events: list[ActivityEvent]
