"""Embedding function factory.

* ``default`` - Chroma's built-in ONNX MiniLM model (good quality, runs locally,
  downloads ~80 MB the first time it is used).
* ``hash``    - a tiny bag-of-words hashing embedder. Zero downloads, so it is used
  by the tests and for offline demos. Retrieval quality is only keyword-level.
"""
import hashlib
import re

import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions


class HashEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            vec = np.zeros(self._dim, dtype=np.float32)
            for token in re.findall(r"\w+", text.lower()):
                bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self._dim
                vec[bucket] += 1.0
            norm = np.linalg.norm(vec)
            if norm:
                vec /= norm
            vectors.append(vec)
        return vectors

    # Chroma uses these to persist/validate the collection's embedder.
    @staticmethod
    def name() -> str:
        return "hash-embedding"

    def default_space(self) -> str:
        return "cosine"

    def get_config(self) -> dict:
        return {"dim": self._dim}

    @staticmethod
    def build_from_config(config: dict) -> "HashEmbeddingFunction":
        return HashEmbeddingFunction(**config)



def build_embedding_function(backend: str):
    if backend == "hash":
        return HashEmbeddingFunction()
    if backend == "default":
        return embedding_functions.DefaultEmbeddingFunction()
    raise ValueError(f"Unknown EMBEDDING_BACKEND '{backend}' (use 'default' or 'hash')")
