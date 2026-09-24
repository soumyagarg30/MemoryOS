from datetime import datetime
from uuid import UUID

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from memoryos.db.base import Base


class ConsolidationSource(Base):
    __tablename__ = "memory_consolidation_sources"
    __table_args__ = (
        CheckConstraint("source_memory_id != consolidated_memory_id", name="distinct_memories"),
    )

    source_memory_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("memories.id", ondelete="RESTRICT"), primary_key=True
    )
    consolidated_memory_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("memories.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
