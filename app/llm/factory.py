import logging

from app.config import Settings
from app.llm.base import BaseLLM
from app.llm.mock_provider import MockLLM
from app.llm.openai_provider import OpenAICompatibleLLM

logger = logging.getLogger(__name__)


def build_llm(settings: Settings) -> BaseLLM:
    provider = settings.llm_provider.lower()
    has_credentials = bool(settings.llm_api_key or settings.llm_base_url)

    if provider == "mock" or (provider == "auto" and not has_credentials):
        logger.warning("Using MOCK LLM (no LLM_API_KEY configured).")
        return MockLLM()

    if provider in ("auto", "openai"):
        logger.info("Using OpenAI-compatible LLM: model=%s base_url=%s",
                    settings.llm_model, settings.llm_base_url or "default")
        return OpenAICompatibleLLM(
            # Local servers such as Ollama need no key, but the SDK requires a non-empty one.
            api_key=settings.llm_api_key or "not-needed",
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
        )

    raise ValueError(f"Unknown LLM_PROVIDER '{settings.llm_provider}' (use auto, openai or mock)")
