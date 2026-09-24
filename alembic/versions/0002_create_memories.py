"""Create the core memories table."""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.create_table(
        "memories",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(), nullable=True),
        sa.Column(
            "memory_type",
            sa.Enum("WORKING", "EPISODIC", "SEMANTIC", "PREFERENCE", "TASK", name="memory_type"),
            nullable=False,
        ),
        sa.Column("importance", sa.Float(), server_default="0.5", nullable=False),
        sa.Column("confidence", sa.Float(), server_default="1.0", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("access_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source", sa.String(255), nullable=True),
        sa.Column(
            "status",
            sa.Enum("ACTIVE", "STALE", "ARCHIVED", "SUPERSEDED", name="memory_status"),
            server_default="ACTIVE",
            nullable=False,
        ),
        sa.Column("superseded_by", sa.Uuid(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), server_default=sa.text("'{}'"), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_memories")),
        sa.ForeignKeyConstraint(
            ["superseded_by"],
            ["memories.id"],
            name=op.f("fk_memories_superseded_by_memories"),
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint("importance >= 0 AND importance <= 1", name="importance_range"),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
        sa.CheckConstraint("access_count >= 0", name="access_count_nonnegative"),
        sa.CheckConstraint("length(trim(content)) > 0", name="content_not_empty"),
        sa.CheckConstraint("superseded_by != id", name="not_self_superseded"),
    )
    op.create_index("ix_memories_user_id_created_at", "memories", ["user_id", "created_at", "id"])
    op.create_index("ix_memories_superseded_by", "memories", ["superseded_by"])


def downgrade() -> None:
    op.drop_table("memories")
    sa.Enum(name="memory_status").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="memory_type").drop(op.get_bind(), checkfirst=True)
