"""Central application settings, loaded from environment variables / .env."""
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", extra="ignore", populate_by_name=True
    )

    log_level: str = "INFO"

    # --- LLM ---
    llm_provider: str = "auto"  # auto | openai | mock
    llm_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY")
    )
    llm_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 60.0

    # --- Vector store / RAG ---
    embedding_backend: str = "default"  # default | hash
    chroma_dir: Path = BASE_DIR / "chroma_db"
    collection_name: str = "knowledge_base"
    docs_dir: Path = BASE_DIR / "data" / "documents"
    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 3

    # --- Chat ---
    max_history_messages: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
