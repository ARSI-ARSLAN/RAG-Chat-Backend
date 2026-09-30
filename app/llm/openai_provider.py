"""Streaming client for any OpenAI-compatible chat API (OpenAI, Groq, Ollama, ...)."""
import logging
from collections.abc import AsyncIterator

from openai import AsyncOpenAI, OpenAIError

from app.llm.base import BaseLLM, LLMError, Messages

logger = logging.getLogger(__name__)


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, api_key: str, model: str, base_url: str | None, timeout: float) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=1)

    async def stream(self, messages: Messages) -> AsyncIterator[str]:
        try:
            response = await self._client.chat.completions.create(
                model=self._model, messages=messages, stream=True, temperature=0.2
            )
            async for event in response:
                if event.choices and event.choices[0].delta.content:
                    yield event.choices[0].delta.content
        except OpenAIError as exc:
            logger.error("LLM request failed: %s", exc)
            raise LLMError(str(exc)) from exc
