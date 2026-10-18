"""ChromaDB index over the memory texts (experiments and lessons).

SQLite stays the source of truth. This index can always be rebuilt from it
(see MemoryStore.reindex), so losing it costs nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import chromadb
from chromadb.config import Settings

import datasets
from memory.embedding import Embedder

DEFAULT_PATH = Path(__file__).resolve().parent / f"chroma{datasets.memory_suffix()}"


class VectorIndex:
    def __init__(self, embedder: Embedder, path: str | Path = DEFAULT_PATH) -> None:
        self.embedder = embedder
        client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        # one collection per embedder, so vectors from different models never mix
        collection_name = "memory-" + re.sub(r"[^A-Za-z0-9._-]", "-", embedder.name)
        self.collection = client.get_or_create_collection(
            collection_name, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, doc_id: str, text: str) -> None:
        self.collection.upsert(ids=[doc_id], documents=[text], embeddings=self.embedder.embed([text]))

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """(doc_id, cosine distance) for the nearest documents, closest first."""
        total = self.collection.count()
        if total == 0 or not query.strip():
            return []
        result = self.collection.query(
            query_embeddings=self.embedder.embed([query]), n_results=min(k, total)
        )
        return list(zip(result["ids"][0], result["distances"][0], strict=True))

    def count(self) -> int:
        return self.collection.count()