"""HTTP routes."""
import asyncio
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.llm.base import LLMError
from app.schemas import ChatRequest
from app.services.chat_service import ChatService
from app.sse import format_sse

logger = logging.getLogger(__name__)
router = APIRouter()


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


@router.get("/health")
async def health(request: Request) -> dict:
    return {"status": "ok", "indexed_chunks": request.app.state.store.count()}


@router.post("/chat", summary="Ask a question; the answer is streamed back as SSE")
async def chat(payload: ChatRequest, service: ChatService = Depends(get_chat_service)):
    # Retrieval happens BEFORE streaming starts, so a failure here can still return a
    # proper HTTP error status. Errors after streaming has begun are sent as SSE events.
    try:
        chunks = await service.retrieve(payload.message)
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=503, detail="Knowledge base is unavailable.") from exc

    async def event_stream() -> AsyncIterator[str]:
        yield format_sse("sources", {"sources": sorted({c.source for c in chunks})})
        try:
            async for token in service.stream_answer(payload.message, chunks, payload.history):
                yield format_sse("token", {"text": token})
            yield format_sse("done", {})
        except LLMError:
            yield format_sse("error", {"message": "The language model is currently unavailable."})
        except asyncio.CancelledError:
            logger.info("Client disconnected mid-stream")
            raise
        except Exception:
            logger.exception("Unexpected error while streaming")
            yield format_sse("error", {"message": "Internal error while generating the response."})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
