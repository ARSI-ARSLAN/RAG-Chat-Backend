"""Request models for the API (validation happens here, not in the routes)."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=2000, description="The user's question.")
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=20,
        description="Optional previous turns, oldest first.",
    )
