from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from memoryos.api.retrieval import get_embedding_service
from memoryos.db.session import DbSession
from memoryos.schemas.archive import ArchiveRequest, DecayRequest, DemoResult, ProvenanceResult
from memoryos.services.archive import provenance, seed_demo
from memoryos.services.decay import DecaySummary, decay_memories
from memoryos.services.embeddings import EmbeddingError, EmbeddingService, EmbeddingTimeout
from memoryos.services.memories import MemoryNotFound
from memoryos.services.storage import MemoryWriteBusy

router = APIRouter(tags=["archive"])


@router.get("/memories/{memory_id}/provenance", response_model=ProvenanceResult)
def get_provenance(memory_id: UUID, db: DbSession):
    try:
        return provenance(db, memory_id)
    except MemoryNotFound as exc:
        raise HTTPException(404, "Memory not found") from exc


@router.post("/memory/decay", response_model=DecaySummary)
def decay(payload: DecayRequest, db: DbSession):
    return decay_memories(db, user_id=payload.user_id, dry_run=payload.dry_run)


@router.post("/demo", response_model=DemoResult)
async def demo(
    payload: ArchiveRequest,
    db: DbSession,
    embeddings: Annotated[EmbeddingService, Depends(get_embedding_service)],
):
    try:
        return await seed_demo(db, payload.user_id, embeddings)
    except EmbeddingTimeout as exc:
        raise HTTPException(504, "Demo embeddings timed out; no demo records were stored") from exc
    except EmbeddingError as exc:
        raise HTTPException(
            502, "Demo embeddings unavailable; no demo records were stored"
        ) from exc
    except MemoryWriteBusy as exc:
        raise HTTPException(409, "Archive is busy; retry shortly") from exc
