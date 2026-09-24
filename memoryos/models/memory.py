from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
    func,
    text,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from memoryos.db.base import Base


class MemoryType(StrEnum):
    WORKING = "WORKING"
    EPISODIC = "EPISODIC"
    SEMANTIC = "SEMANTIC"
    PREFERENCE = "PREFERENCE"
    TASK = "TASK"


class MemoryStatus(StrEnum):
    ACTIVE = "ACTIVE"
    STALE = "STALE"
    ARCHIVED = "ARCHIVED"
    SUPERSEDED = "SUPERSEDED"


class Memory(Base):
    __tablename__ = "memories"
    __table_args__ = (
        CheckConstraint("importance >= 0 AND importance <= 1", name="importance_range"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
        CheckConstraint("access_count >= 0", name="access_count_nonnegative"),
        CheckConstraint("length(trim(content)) > 0", name="content_not_empty"),
        CheckConstraint("superseded_by != id", name="not_self_superseded"),
        Index("ix_memories_user_id_created_at", "user_id", "created_at", "id"),
        Index("ix_memories_superseded_by", "superseded_by"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(), nullable=True)
    memory_type: Mapped[MemoryType] = mapped_column(
        SAEnum(MemoryType, name="memory_type"), nullable=False
    )
    importance: Mapped[float] = mapped_column(Float, default=0.5, server_default="0.5")
    confidence: Mapped[float] = mapped_column(Float, default=1.0, server_default="1.0")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    access_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[MemoryStatus] = mapped_column(
        SAEnum(MemoryStatus, name="memory_status"),
        default=MemoryStatus.ACTIVE,
        server_default="ACTIVE",
    )
    superseded_by: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("memories.id", ondelete="RESTRICT")
    )
    consolidated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, default=dict, server_default=text("'{}'"), nullable=False
    )
