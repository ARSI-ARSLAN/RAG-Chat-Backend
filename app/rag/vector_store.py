"""Thin wrapper around a persistent Chroma collection."""
import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    source: str


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    distance: float  # cosine distance: lower = more similar


class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str, embedding_function) -> None:
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def count(self) -> int:
        return self._collection.count()

    def replace_source(self, source: str, chunks: list[DocumentChunk]) -> None:
        """Idempotently (re)store all chunks of one source file."""
        self._collection.delete(where={"source": source})
        if not chunks:
            return
        self._collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source} for c in chunks],
        )

    def query(self, text: str, k: int) -> list[RetrievedChunk]:
        total = self.count()
        if total == 0:
            return []
        result = self._collection.query(query_texts=[text], n_results=min(k, total))
        return [
            RetrievedChunk(text=doc, source=meta["source"], distance=dist)
            for doc, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
