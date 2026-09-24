"""Add consolidation markers and durable source provenance."""

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "memories", sa.Column("consolidated_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.create_table(
        "memory_consolidation_sources",
        sa.Column("source_memory_id", sa.Uuid(), nullable=False),
        sa.Column("consolidated_memory_id", sa.Uuid(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("source_memory_id", name=op.f("pk_memory_consolidation_sources")),
        sa.ForeignKeyConstraint(
            ["source_memory_id"],
            ["memories.id"],
            ondelete="RESTRICT",
            name=op.f("fk_memory_consolidation_sources_source_memory_id_memories"),
        ),
        sa.ForeignKeyConstraint(
            ["consolidated_memory_id"],
            ["memories.id"],
            ondelete="RESTRICT",
            name=op.f("fk_memory_consolidation_sources_consolidated_memory_id_memories"),
        ),
        sa.CheckConstraint("source_memory_id != consolidated_memory_id", name="distinct_memories"),
    )
    op.create_index(
        "ix_memory_consolidation_sources_consolidated_memory_id",
        "memory_consolidation_sources",
        ["consolidated_memory_id"],
    )


def downgrade() -> None:
    op.drop_table("memory_consolidation_sources")
    op.drop_column("memories", "consolidated_at")
