"""Find the chunks that best match a question: BM25 word matching, optionally fused with embeddings.

BM25 is the classic sparse ranking function. It is exact, explainable and needs no model download, which
suits a corpus full of identifiers (`catboost`, `psi_major`, `id_31`) that embeddings blur. When an embedder is
supplied, its ranking is merged with BM25 by reciprocal rank fusion, so meaning-based matches can surface too.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from research.corpus import Chunk

STOP = frozenset("a an and are as at be by for from has have how i in is it of on or that the this to was we what when where which who why with do does did can".split())


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[a-z0-9_]+", text.lower()):
        if token in STOP:
            continue
        tokens.append(token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token)
    return tokens


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder=None, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks, self.k1, self.b, self.embedder = chunks, k1, b, embedder
        self.tf = [Counter(tokenize(c.text + " " + c.section)) for c in chunks]
        self.lengths = [sum(tf.values()) for tf in self.tf]
        self.avg = (sum(self.lengths) / len(self.lengths)) if chunks else 0.0
        df: Counter = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.vectors = embedder.embed([c.text for c in chunks]) if embedder and chunks else None

    def bm25(self, query: str) -> list[float]:
        terms = tokenize(query)
        scores = []
        for tf, length in zip(self.tf, self.lengths, strict=True):
            score = 0.0
            for term in terms:
                if term in tf:
                    freq = tf[term]
                    score += self.idf[term] * freq * (self.k1 + 1) / (freq + self.k1 * (1 - self.b + self.b * length / (self.avg or 1)))
            scores.append(score)
        return scores

    def search(self, query: str, k: int = 5, min_score: float = 0.0) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        sparse = self.bm25(query)
        if self.vectors is not None:
            q = self.embedder.embed([query])[0]
            dense = [sum(a * b for a, b in zip(q, v, strict=True)) for v in self.vectors]
            fused = [0.0] * len(self.chunks)
            for ranking in (sparse, dense):
                for rank, idx in enumerate(sorted(range(len(ranking)), key=lambda i: ranking[i], reverse=True)[:50], start=1):
                    if ranking[idx] > 0:
                        fused[idx] += 1.0 / (60 + rank)
            order = sorted(range(len(fused)), key=lambda i: fused[i], reverse=True)
            keep = [i for i in order if sparse[i] > min_score or fused[i] > 0][:k]
            return [(self.chunks[i], fused[i]) for i in keep if fused[i] > 0]
        order = sorted(range(len(sparse)), key=lambda i: sparse[i], reverse=True)
        return [(self.chunks[i], sparse[i]) for i in order[:k] if sparse[i] > min_score]
