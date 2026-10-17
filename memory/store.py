"""Persistent memory of experiments and lessons (SQLite + FTS5 full-text search).

The agents will use this to avoid repeating work: before proposing an experiment
they search for similar past ones, and after running one they record the result.
Facts (config, metrics, verdict) are stored structured; free text is indexed for
search. Everything recorded here is a verified result, never a guess.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory.vector import VectorIndex

DEFAULT_PATH = Path(__file__).resolve().parent / "autoevolve.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    hypothesis TEXT,
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    dataset_sha256 TEXT NOT NULL DEFAULT '',
    git_commit TEXT,
    git_dirty INTEGER,
    metrics_json TEXT NOT NULL,
    verdict TEXT,
    conclusion TEXT,
    mlflow_run_id TEXT,
    UNIQUE (kind, name, config_hash, dataset_sha256)
);
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    experiment_id INTEGER REFERENCES experiments(id),
    text TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '',
    UNIQUE (experiment_id, text)
);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    text, ref_table UNINDEXED, ref_id UNINDEXED
);
"""


def config_hash(config: dict) -> str:
    """Stable fingerprint of a configuration, so repeats can be detected."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _fts_query(text: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return " OR ".join(f'"{token}"' for token in tokens)


def _experiment_text(row: sqlite3.Row) -> str:
    return " ".join(
        [row["kind"], row["name"], row["hypothesis"] or "", row["verdict"] or "",
         row["conclusion"] or "", row["config_json"]]
    )


class MemoryStore:
    def __init__(self, path: str | Path = DEFAULT_PATH, vectors: VectorIndex | None = None) -> None:
        self.path = str(path)
        self.vectors = vectors
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.executescript(SCHEMA)
        except sqlite3.OperationalError as exc:
            raise RuntimeError(f"SQLite FTS5 is required but unavailable: {exc}") from exc

    def close(self) -> None:
        self.db.close()

    # ---- writing -------------------------------------------------------
    def record_experiment(
        self,
        kind: str,
        name: str,
        config: dict,
        metrics: dict,
        *,
        hypothesis: str = "",
        verdict: str = "",
        conclusion: str = "",
        dataset_sha256: str = "",
        git_commit: str = "",
        git_dirty: bool | None = None,
        mlflow_run_id: str = "",
    ) -> tuple[int, bool]:
        """Store an experiment. Returns (id, created); re-recording the same one is a no-op."""
        cfg_hash = config_hash(config)
        cursor = self.db.execute(
            """INSERT OR IGNORE INTO experiments
               (created_at, kind, name, hypothesis, config_json, config_hash, dataset_sha256,
                git_commit, git_dirty, metrics_json, verdict, conclusion, mlflow_run_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.now(UTC).isoformat(timespec="seconds"),
                kind,
                name,
                hypothesis,
                json.dumps(config, sort_keys=True, default=str),
                cfg_hash,
                dataset_sha256,
                git_commit,
                None if git_dirty is None else int(git_dirty),
                json.dumps(metrics, sort_keys=True, default=str),
                verdict,
                conclusion,
                mlflow_run_id,
            ),
        )
        created = cursor.rowcount == 1
        row = self.db.execute(
            "SELECT id FROM experiments WHERE kind=? AND name=? AND config_hash=? AND dataset_sha256=?",
            (kind, name, cfg_hash, dataset_sha256),
        ).fetchone()
        if created:
            stored = self.db.execute("SELECT * FROM experiments WHERE id=?", (row["id"],)).fetchone()
            text = _experiment_text(stored)
            self.db.execute(
                "INSERT INTO memory_fts (text, ref_table, ref_id) VALUES (?, 'experiments', ?)",
                (text, row["id"]),
            )
            if self.vectors:
                self.vectors.upsert(f"experiments:{row['id']}", text)
        self.db.commit()
        return row["id"], created

    def add_lesson(self, text: str, *, experiment_id: int | None = None, tags: str = "") -> tuple[int, bool]:
        cursor = self.db.execute(
            "INSERT OR IGNORE INTO lessons (created_at, experiment_id, text, tags) VALUES (?,?,?,?)",
            (datetime.now(UTC).isoformat(timespec="seconds"), experiment_id, text, tags),
        )
        created = cursor.rowcount == 1
        row = self.db.execute(
            "SELECT id FROM lessons WHERE experiment_id IS ? AND text=?", (experiment_id, text)
        ).fetchone()
        if created:
            self.db.execute(
                "INSERT INTO memory_fts (text, ref_table, ref_id) VALUES (?, 'lessons', ?)",
                (f"{text} {tags}", row["id"]),
            )
            if self.vectors:
                self.vectors.upsert(f"lessons:{row['id']}", f"{text} {tags}")
        self.db.commit()
        return row["id"], created

    def reindex(self) -> int:
        """Rebuild the vector index from SQLite (after adding or changing an embedder)."""
        if not self.vectors:
            raise RuntimeError("no vector index attached")
        count = 0
        for row in self.db.execute("SELECT * FROM experiments").fetchall():
            self.vectors.upsert(f"experiments:{row['id']}", _experiment_text(row))
            count += 1
        for row in self.db.execute("SELECT * FROM lessons").fetchall():
            self.vectors.upsert(f"lessons:{row['id']}", f"{row['text']} {row['tags']}")
            count += 1
        return count

    # ---- reading -------------------------------------------------------
    def get_experiment(self, experiment_id: int) -> dict | None:
        row = self.db.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        return self._experiment(row) if row else None

    def already_tried(self, kind: str, name: str, config: dict, dataset_sha256: str = "") -> dict | None:
        """The earlier result for exactly this experiment on this data, if any."""
        row = self.db.execute(
            "SELECT * FROM experiments WHERE kind=? AND name=? AND config_hash=? AND dataset_sha256=?",
            (kind, name, config_hash(config), dataset_sha256),
        ).fetchone()
        return self._experiment(row) if row else None

    def recent(self, kind: str | None = None, limit: int = 10) -> list[dict]:
        if kind:
            rows = self.db.execute(
                "SELECT * FROM experiments WHERE kind=? ORDER BY id DESC LIMIT ?", (kind, limit)
            ).fetchall()
        else:
            rows = self.db.execute("SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._experiment(r) for r in rows]

    def lessons(self, limit: int = 50) -> list[dict]:
        rows = self.db.execute("SELECT * FROM lessons ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Hybrid search: keyword (SQLite FTS) and meaning (ChromaDB), merged by rank.

        Each ranking gives a document 1 / (60 + rank) points (reciprocal rank fusion), so
        documents found by both methods rise to the top. Without a vector index it is
        keyword-only.
        """
        rankings: dict[str, list[tuple[str, int]]] = {"keyword": self._keyword_hits(query, limit * 3)}
        if self.vectors:
            rankings["vector"] = self._vector_hits(query, limit * 3)

        scores: dict[tuple[str, int], float] = {}
        found_by: dict[tuple[str, int], list[str]] = {}
        for method, ranking in rankings.items():
            for rank, key in enumerate(ranking):
                scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank + 1)
                found_by.setdefault(key, []).append(method)

        results = []
        for key in sorted(scores, key=scores.get, reverse=True)[:limit]:
            table, ref_id = key
            if table == "experiments":
                item = self.get_experiment(ref_id)
            else:
                row = self.db.execute("SELECT * FROM lessons WHERE id=?", (ref_id,)).fetchone()
                item = dict(row) if row else None
            if item:
                results.append({"type": table, "score": scores[key], "found_by": found_by[key], "item": item})
        return results

    def _keyword_hits(self, query: str, limit: int) -> list[tuple[str, int]]:
        match = _fts_query(query)
        if not match:
            return []
        rows = self.db.execute(
            "SELECT ref_table, ref_id FROM memory_fts WHERE memory_fts MATCH ? ORDER BY bm25(memory_fts) LIMIT ?",
            (match, limit),
        ).fetchall()
        return [(row["ref_table"], int(row["ref_id"])) for row in rows]

    def _vector_hits(self, query: str, limit: int) -> list[tuple[str, int]]:
        hits = []
        for doc_id, _distance in self.vectors.search(query, limit):
            table, _, ref_id = doc_id.partition(":")
            hits.append((table, int(ref_id)))
        return hits

    @staticmethod
    def _experiment(row: sqlite3.Row) -> dict:
        item = dict(row)
        item["config"] = json.loads(item.pop("config_json"))
        item["metrics"] = json.loads(item.pop("metrics_json"))
        return item