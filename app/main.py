"""FastAPI application factory."""
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import router
from app.config import Settings, get_settings
from app.llm.factory import build_llm
from app.logging_config import configure_logging
from app.rag.ingest import build_store, ingest_directory
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = build_store(settings)
        if store.count() == 0:
            logger.info("Vector store is empty - ingesting sample documents")
            ingest_directory(store, settings)
        logger.info("Vector store ready: %d chunks", store.count())

        app.state.store = store
        app.state.chat_service = ChatService(
            store, build_llm(settings), settings.top_k, settings.max_history_messages
        )
        yield

    app = FastAPI(title="RAG Chat Backend", version="1.0.0", lifespan=lifespan)
    app.include_router(router)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = uuid.uuid4().hex[:8]
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("[%s] %s %s -> %d (%.0f ms to first byte)",
                    request_id, request.method, request.url.path, response.status_code, elapsed_ms)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    return app


app = create_app()
