from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from memoryos.db.session import DbSession
from memoryos.schemas.consolidation import ConsolidationRequest, ConsolidationResult
from memoryos.services.consolidation import (
    ConsolidationBusy,
    ConsolidationDatabaseError,
    ConsolidationOutputError,
    ConsolidationRefused,
    MemoryConsolidationService,
)
from memoryos.services.embeddings import (
    EmbeddingBusy,
    EmbeddingError,
    EmbeddingService,
    EmbeddingTimeout,
)
from memoryos.services.ollama import APIError, APITimeoutError, RateLimitError

router = APIRouter(prefix="/memory", tags=["memory consolidation"])


def get_consolidation_service(request: Request) -> MemoryConsolidationService:
    client = request.app.state.llm_client
    if client is None:
        raise HTTPException(503, "Memory consolidation is not configured")
    settings = request.app.state.settings
    return MemoryConsolidationService(
        client,
        EmbeddingService(client, settings.embedding_model, settings.embedding_dimensions),
        model=settings.consolidation_model,
        similarity=settings.consolidation_similarity,
        min_confidence=settings.consolidation_min_confidence,
    )


@router.post("/consolidate", response_model=ConsolidationResult)
async def consolidate_memories(
    payload: ConsolidationRequest,
    db: DbSession,
    service: Annotated[MemoryConsolidationService, Depends(get_consolidation_service)],
) -> ConsolidationResult:
    try:
        return await service.consolidate(db, payload.user_id, payload.after_id)
    except (APITimeoutError, EmbeddingTimeout) as exc:
        raise HTTPException(504, "Memory consolidation timed out") from exc
    except (RateLimitError, EmbeddingBusy, ConsolidationDatabaseError) as exc:
        raise HTTPException(503, "Memory consolidation is temporarily unavailable") from exc
    except ConsolidationBusy as exc:
        raise HTTPException(
            409, "Memory state changed or another write is in progress; retry"
        ) from exc
    except ConsolidationRefused as exc:
        raise HTTPException(422, "The model declined to consolidate these memories") from exc
    except (APIError, EmbeddingError, ConsolidationOutputError) as exc:
        raise HTTPException(
            502, "Memory consolidation model returned an invalid result or failed"
        ) from exc
