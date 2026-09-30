"""Load documents from disk, chunk them, and store them in the vector DB.

Run manually (re-indexes everything):
    python -m app.rag.ingest
"""
import logging
from pathlib import Path

from app.config import Settings, get_settings
from app.logging_config import configure_logging
from app.rag.chunking import chunk_text
from app.rag.embeddings import build_embedding_function
from app.rag.vector_store import DocumentChunk, VectorStore

logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".txt", ".md"}


def load_chunks(path: Path, chunk_size: int, overlap: int) -> list[DocumentChunk]:
    text = path.read_text(encoding="utf-8")
    return [
        DocumentChunk(id=f"{path.name}::{i}", text=piece, source=path.name)
        for i, piece in enumerate(chunk_text(text, chunk_size, overlap))
    ]


def ingest_directory(store: VectorStore, settings: Settings) -> int:
    """Index every .txt/.md file in the docs folder. Returns the number of chunks stored."""
    files = sorted(p for p in settings.docs_dir.glob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        logger.warning("No documents found in %s", settings.docs_dir)
        return 0

    total = 0
    for path in files:
        chunks = load_chunks(path, settings.chunk_size, settings.chunk_overlap)
        store.replace_source(path.name, chunks)
        logger.info("Ingested %s (%d chunks)", path.name, len(chunks))
        total += len(chunks)
    return total


def build_store(settings: Settings) -> VectorStore:
    return VectorStore(
        settings.chroma_dir,
        settings.collection_name,
        build_embedding_function(settings.embedding_backend),
    )


if __name__ == "__main__":
    cfg = get_settings()
    configure_logging(cfg.log_level)
    n = ingest_directory(build_store(cfg), cfg)
    logger.info("Done. %d chunks indexed.", n)
