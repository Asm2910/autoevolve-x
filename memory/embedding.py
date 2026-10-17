"""Turn text into vectors for similarity search.

Two embedders, same interface:
  - MiniLMEmbedder: a small real language-embedding model (ONNX, runs on CPU). It is
    downloaded once (~80 MB) on first use. This is the default for real use.
  - HashingEmbedder: deterministic word hashing, no download. It only matches on shared
    words, so it is for tests and offline use, not for semantic search.

Each embedder has a `name`. The vector index keeps one collection per embedder name,
so vectors from different models are never mixed.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    name = "hashing-256"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in re.findall(r"[a-z0-9_]+", text.lower()):
                digest = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
                vec[digest % self.dim] += 1.0 if (digest >> 63) & 1 else -1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class MiniLMEmbedder:
    name = "minilm-onnx"

    def __init__(self) -> None:
        self._function = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._function is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            self._function = DefaultEmbeddingFunction()  # downloads the model on first use
        return [[float(x) for x in vector] for vector in self._function(texts)]


def get_embedder(name: str) -> Embedder:
    if name == "minilm":
        return MiniLMEmbedder()
    if name == "hashing":
        return HashingEmbedder()
    raise ValueError(f"unknown embedder '{name}' (use 'minilm' or 'hashing')")