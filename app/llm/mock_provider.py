"""Offline fallback LLM so the whole pipeline can be demoed without an API key.

It simply echoes the most relevant retrieved passage, word by word.
"""
import asyncio
from collections.abc import AsyncIterator

from app.llm.base import BaseLLM, Messages
from app.services.prompts import CONTEXT_MARKER


class MockLLM(BaseLLM):
    async def stream(self, messages: Messages) -> AsyncIterator[str]:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        context = system.split(CONTEXT_MARKER, 1)[-1].strip()
        first_passage = context.split("\n\n---\n\n")[0][:400]
        reply = (
            "[mock LLM - set LLM_API_KEY for real answers] "
            f"Most relevant passage I found: {first_passage}"
        )
        for word in reply.split(" "):
            yield word + " "
            await asyncio.sleep(0.02)
