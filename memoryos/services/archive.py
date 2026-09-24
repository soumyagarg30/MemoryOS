from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from memoryos.db.locks import try_lock_user
from memoryos.models import Memory
from memoryos.models.consolidation import ConsolidationSource
from memoryos.schemas.archive import DemoResult, ProvenanceResult
from memoryos.schemas.chat import ActivityEvent
from memoryos.schemas.memory import MemoryResponse
from memoryos.services.embeddings import EmbeddingService
from memoryos.services.memories import get_memory
from memoryos.services.storage import MemoryWriteBusy

DEMO_SOURCE = "memoryos-demo-v1"
DEMO_RECORDS = [
    ("Uses MongoDB for new projects.", "PREFERENCE"),
    ("Prefers Python for data analysis at work.", "PREFERENCE"),
    ("Prefers C++ for competitive programming practice.", "PREFERENCE"),
    ("Practiced graph traversal algorithms, focusing on BFS and DFS, on Monday.", "EPISODIC"),
    ("Practiced graph traversal algorithms, focusing on BFS and DFS, on Tuesday.", "EPISODIC"),
    ("Practiced graph traversal algorithms, focusing on BFS and DFS, on Wednesday.", "EPISODIC"),
]


def provenance(db: Session, memory_id: UUID) -> ProvenanceResult:
    memory = get_memory(db, memory_id)

    def records(statement):
        return [
            MemoryResponse.model_validate(m)
            for m in db.scalars(
                statement.where(Memory.user_id == memory.user_id).order_by(Memory.id)
            )
        ]

    sources = records(
        select(Memory)
        .join(ConsolidationSource, ConsolidationSource.source_memory_id == Memory.id)
        .where(ConsolidationSource.consolidated_memory_id == memory.id)
    )
    targets = records(
        select(Memory)
        .join(ConsolidationSource, ConsolidationSource.consolidated_memory_id == Memory.id)
        .where(ConsolidationSource.source_memory_id == memory.id)
    )
    replacement = db.get(Memory, memory.superseded_by) if memory.superseded_by else None
    return ProvenanceResult(
        memory_id=memory.id,
        derived_from=sources,
        consolidated_into=targets,
        supersedes=records(select(Memory).where(Memory.superseded_by == memory.id)),
        superseded_by=MemoryResponse.model_validate(replacement)
        if replacement and replacement.user_id == memory.user_id
        else None,
    )


def demo_records(db: Session, user_id: UUID) -> list[MemoryResponse]:
    return [
        MemoryResponse.model_validate(m)
        for m in db.scalars(
            select(Memory)
            .where(Memory.user_id == user_id, Memory.source == DEMO_SOURCE)
            .order_by(Memory.id)
        )
    ]


def persist_demo(db: Session, user_id: UUID, vectors: list[list[float]]) -> DemoResult:
    try:
        if not try_lock_user(db, user_id):
            raise MemoryWriteBusy
        existing = demo_records(db, user_id)
        if existing:
            db.rollback()
            return DemoResult(memories=existing, already_loaded=True, memory_activity_events=[])
        for (content, memory_type), embedding in zip(DEMO_RECORDS, vectors, strict=True):
            db.add(
                Memory(
                    user_id=user_id,
                    content=content,
                    memory_type=memory_type,
                    embedding=embedding,
                    importance=0.8,
                    confidence=0.95,
                    source=DEMO_SOURCE,
                    metadata_={"synthetic_demo": True},
                )
            )
        db.flush()
        memories = demo_records(db, user_id)
        db.commit()
        return DemoResult(
            memories=memories,
            already_loaded=False,
            memory_activity_events=[
                ActivityEvent(
                    kind="DEMO_SEEDED",
                    message="Six synthetic archive records stored",
                    memory_ids=[m.id for m in memories],
                )
            ],
        )
    except Exception:
        db.rollback()
        raise


async def seed_demo(db: Session, user_id: UUID, embeddings: EmbeddingService) -> DemoResult:
    existing = await run_in_threadpool(demo_records, db, user_id)
    await run_in_threadpool(db.rollback)
    if existing:
        return DemoResult(memories=existing, already_loaded=True, memory_activity_events=[])
    vectors = [await embeddings.embed(content) for content, _ in DEMO_RECORDS]
    return await run_in_threadpool(persist_demo, db, user_id, vectors)
