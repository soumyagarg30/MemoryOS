from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from memoryos.db.session import DbSession
from memoryos.models.memory import MemoryStatus, MemoryType
from memoryos.schemas.memory import MemoryCreate, MemoryResponse
from memoryos.services import memories as service
from memoryos.services.conflicts import (
    ConflictDetectionBusy,
    ConflictDetectionError,
    ConflictDetectionRefused,
    ConflictDetectionTimeout,
    MemoryConflictDetector,
)
from memoryos.services.embeddings import (
    EmbeddingBusy,
    EmbeddingError,
    EmbeddingRejected,
    EmbeddingService,
    EmbeddingTimeout,
)
from memoryos.services.storage import (
    InvalidMemoryEmbedding,
    MemoryStorageError,
    MemoryStorageService,
    MemoryWriteBusy,
)

router = APIRouter(prefix="/memories", tags=["memories"])


def get_storage_service(request: Request) -> MemoryStorageService:
    client = request.app.state.llm_client
    if client is None:
        raise HTTPException(503, "Memory storage requires a configured Ollama client")
    settings = request.app.state.settings
    return MemoryStorageService(
        EmbeddingService(client, settings.embedding_model, settings.embedding_dimensions),
        MemoryConflictDetector(client, settings.conflict_model),
    )


@router.post("", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_memory(
    payload: MemoryCreate,
    db: DbSession,
    response: Response,
    storage: Annotated[MemoryStorageService, Depends(get_storage_service)],
) -> MemoryResponse:
    try:
        result = await storage.store(db, payload)
    except (EmbeddingTimeout, ConflictDetectionTimeout) as exc:
        raise HTTPException(504, "Memory storage model request timed out") from exc
    except (EmbeddingBusy, ConflictDetectionBusy, MemoryStorageError) as exc:
        raise HTTPException(503, "Memory storage is temporarily unavailable") from exc
    except (EmbeddingRejected, ConflictDetectionRefused) as exc:
        raise HTTPException(422, "The model declined to process this memory") from exc
    except (EmbeddingError, ConflictDetectionError) as exc:
        raise HTTPException(
            502, "Memory storage model returned an invalid result or failed"
        ) from exc
    except InvalidMemoryEmbedding as exc:
        raise HTTPException(422, "Memory embedding must have a nonzero norm") from exc
    except MemoryWriteBusy as exc:
        raise HTTPException(
            409, "Memory state changed or another write is in progress; retry"
        ) from exc
    except service.InvalidSupersedingMemory as exc:
        raise HTTPException(422, "superseded_by must reference a memory for the same user") from exc
    except service.MemoryConflict as exc:
        raise HTTPException(409, "Memory conflicts with existing data") from exc
    response.status_code = 201 if result.created else 200
    response.headers["Location"] = f"/memories/{result.memory.id}"
    return result.memory


@router.get("", response_model=list[MemoryResponse])
def list_memories(
    db: DbSession,
    user_id: UUID | None = None,
    memory_type: MemoryType | None = None,
    status: MemoryStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[MemoryResponse]:
    memories = service.list_memories(
        db, user_id=user_id, memory_type=memory_type, status=status, limit=limit, offset=offset
    )
    return [MemoryResponse.model_validate(memory) for memory in memories]


@router.get("/{memory_id}", response_model=MemoryResponse)
def get_memory(memory_id: UUID, db: DbSession) -> MemoryResponse:
    try:
        memory = service.get_memory(db, memory_id)
    except service.MemoryNotFound as exc:
        raise HTTPException(404, "Memory not found") from exc
    return MemoryResponse.model_validate(memory)


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_memory(memory_id: UUID, db: DbSession) -> Response:
    try:
        service.delete_memory(db, memory_id)
    except service.MemoryNotFound as exc:
        raise HTTPException(404, "Memory not found") from exc
    except service.MemoryConflict as exc:
        raise HTTPException(409, "Memory is referenced by another memory") from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
