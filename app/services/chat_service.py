"""Orchestrates the RAG flow: retrieve context -> build prompt -> stream LLM answer."""
import asyncio
import logging
from collections.abc import AsyncIterator

from app.llm.base import BaseLLM
from app.rag.vector_store import RetrievedChunk, VectorStore
from app.schemas import ChatMessage
from app.services.prompts import build_messages

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, store: VectorStore, llm: BaseLLM, top_k: int, max_history: int) -> None:
        self._store = store
        self._llm = llm
        self._top_k = top_k
        self._max_history = max_history

    async def retrieve(self, question: str) -> list[RetrievedChunk]:
        # Chroma is synchronous and embedding is CPU-bound, so keep it off the event loop.
        chunks = await asyncio.to_thread(self._store.query, question, self._top_k)
        logger.info(
            "Retrieved %d chunks %s",
            len(chunks),
            [(c.source, round(c.distance, 3)) for c in chunks],
        )
        return chunks

    def stream_answer(
        self, question: str, chunks: list[RetrievedChunk], history: list[ChatMessage]
    ) -> AsyncIterator[str]:
        messages = build_messages(question, chunks, history[-self._max_history :])
        return self._llm.stream(messages)
