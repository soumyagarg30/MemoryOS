from fastapi import APIRouter

from memoryos.api.archive import router as archive_router
from memoryos.api.chat import router as chat_router
from memoryos.api.consolidation import router as consolidation_router
from memoryos.api.extraction import router as extraction_router
from memoryos.api.health import router as health_router
from memoryos.api.memories import router as memories_router
from memoryos.api.retrieval import router as retrieval_router

api_router = APIRouter()
api_router.include_router(archive_router)
api_router.include_router(chat_router)
api_router.include_router(health_router)
api_router.include_router(memories_router)
api_router.include_router(extraction_router)
api_router.include_router(retrieval_router)
api_router.include_router(consolidation_router)
