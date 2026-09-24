from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from memoryos.db.session import DbSession
from memoryos.schemas.retrieval import MemoryRetrievalRequest, MemoryRetrievalResult
from memoryos.services.embeddings import (
    EmbeddingBusy,
    EmbeddingError,
    EmbeddingRejected,
    EmbeddingService,
    EmbeddingTimeout,
)
from memoryos.services.retrieval import RetrievalError, retrieve_memories

router = APIRouter(prefix="/memory", tags=["memory retrieval"])


def get_embedding_service(request: Request) -> EmbeddingService:
    client = request.app.state.llm_client
    if client is None:
        raise HTTPException(503, "Memory retrieval is not configured")
    settings = request.app.state.settings
    return EmbeddingService(client, settings.embedding_model, settings.embedding_dimensions)


@router.post("/retrieve", response_model=MemoryRetrievalResult, response_model_exclude_unset=True)
async def retrieve_memory(
    payload: MemoryRetrievalRequest,
    db: DbSession,
    embeddings: Annotated[EmbeddingService, Depends(get_embedding_service)],
) -> MemoryRetrievalResult:
    try:
        vector = await embeddings.embed(payload.query)
    except EmbeddingTimeout as exc:
        raise HTTPException(504, "Query embedding timed out") from exc
    except EmbeddingBusy as exc:
        raise HTTPException(503, "Query embedding is temporarily unavailable") from exc
    except EmbeddingRejected as exc:
        raise HTTPException(422, "The embedding provider rejected the query") from exc
    except EmbeddingError as exc:
        raise HTTPException(502, "Query embedding provider failed") from exc
    try:
        return await run_in_threadpool(
            retrieve_memories, db, payload.user_id, vector, payload.limit, payload.debug
        )
    except RetrievalError as exc:
        raise HTTPException(503, "Memory retrieval is temporarily unavailable") from exc
