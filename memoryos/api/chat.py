from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from memoryos.api.memories import get_storage_service
from memoryos.db.session import DbSession
from memoryos.schemas.chat import ChatRequest, ChatResult
from memoryos.services.chat import ChatService
from memoryos.services.extraction import MemoryExtractionService
from memoryos.services.ollama import APIError, APITimeoutError, RateLimitError
from memoryos.services.storage import MemoryStorageService

router = APIRouter(tags=["chat"])


def get_chat_service(
    request: Request, storage: Annotated[MemoryStorageService, Depends(get_storage_service)]
) -> ChatService:
    settings = request.app.state.settings
    client = request.app.state.llm_client
    return ChatService(
        client,
        storage,
        MemoryExtractionService(client, settings.extraction_model),
        settings.chat_model,
    )


@router.post("/chat", response_model=ChatResult)
async def chat(
    payload: ChatRequest, db: DbSession, service: Annotated[ChatService, Depends(get_chat_service)]
) -> ChatResult:
    try:
        return await service.chat(db, payload)
    except APITimeoutError as exc:
        raise HTTPException(
            504, "The assistant timed out. No new chat memories were stored."
        ) from exc
    except RateLimitError as exc:
        raise HTTPException(503, "The assistant is busy. Try again shortly.") from exc
    except (APIError, ValueError) as exc:
        raise HTTPException(
            502, "The assistant is unavailable or returned an invalid response."
        ) from exc
