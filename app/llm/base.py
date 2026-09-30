"""Common interface so the rest of the app doesn't care which LLM is used."""
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

Messages = list[dict[str, str]]  # [{"role": "system"|"user"|"assistant", "content": "..."}]


class LLMError(Exception):
    """Raised when the language model cannot produce a response."""


class BaseLLM(ABC):
    @abstractmethod
    def stream(self, messages: Messages) -> AsyncIterator[str]:
        """Yield the answer piece by piece."""
