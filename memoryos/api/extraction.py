from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from memoryos.schemas.extraction import MemoryExtractionRequest, MemoryExtractionResult
from memoryos.services.extraction import (
    ExtractionBusy,
    ExtractionError,
    ExtractionRefused,
    ExtractionTimeout,
    InvalidExtractionOutput,
    MemoryExtractionService,
)
from memoryos.services.ollama import OllamaClient

router = APIRouter(prefix="/memory", tags=["memory extraction"])


def get_llm_client(request: Request) -> OllamaClient:
    client = request.app.state.llm_client
    if client is None:
        raise HTTPException(503, "Memory extraction is not configured")
    return client


@router.post("/extract", response_model=MemoryExtractionResult)
async def extract_memory(
    payload: MemoryExtractionRequest,
    request: Request,
    client: Annotated[OllamaClient, Depends(get_llm_client)],
) -> MemoryExtractionResult:
    service = MemoryExtractionService(client, request.app.state.settings.extraction_model)
    try:
        return await service.extract(payload.message)
    except ExtractionTimeout as exc:
        raise HTTPException(504, "Memory extraction timed out") from exc
    except ExtractionBusy as exc:
        raise HTTPException(503, "Memory extraction is temporarily unavailable") from exc
    except ExtractionRefused as exc:
        raise HTTPException(
            422, "The model declined to extract memories from this message"
        ) from exc
    except InvalidExtractionOutput as exc:
        raise HTTPException(502, "The model returned an invalid extraction result") from exc
    except ExtractionError as exc:
        raise HTTPException(502, "Memory extraction provider failed") from exc
