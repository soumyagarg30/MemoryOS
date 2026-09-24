from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from memoryos.api.errors import validation_error_handler
from memoryos.api.router import api_router
from memoryos.config import Settings, get_settings
from memoryos.db.session import create_db_engine, create_session_factory
from memoryos.services.ollama import OllamaClient


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_db_engine(settings)
        app.state.session_factory = create_session_factory(engine)
        app.state.settings = settings
        app.state.llm_client = None
        try:
            app.state.llm_client = OllamaClient(
                base_url=str(settings.ollama_base_url), timeout=settings.ollama_timeout
            )
            yield
        finally:
            try:
                if app.state.llm_client is not None:
                    await app.state.llm_client.close()
            finally:
                engine.dispose()

    app = FastAPI(title="MemoryOS", version="0.1.0", debug=settings.debug, lifespan=lifespan)
    app.add_exception_handler(RequestValidationError, validation_error_handler)

    @app.exception_handler(SQLAlchemyError)
    async def database_error_handler(request, exc):
        return JSONResponse(status_code=503, content={"detail": "Database temporarily unavailable"})

    app.include_router(api_router)
    return app


app = create_app()
