"""What the research agent may read: the project's own documents, reports and memory. Nothing from the web.

Markdown files are cut at headings (and long sections at paragraph breaks) into chunks that each remember
their source file and section. Lessons and recent experiments come from the memory store. The sealed test
split is data, not text, and is never part of this corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MAX_CHARS = 900
MIN_CHARS = 40
DOCS = ("README.md", "SECURITY.md", "docs/*.md")
REPORTS = ("arena_report.md", "judge_report.md", "evaluation_report.md", "data_review_report.md",
           "mission_report.md", "model_report.md", "ray_benchmark.md")


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    section: str
    text: str
    kind: str  # doc | report | lesson | experiment


def chunk_markdown(source: str, text: str, kind: str = "doc") -> list[Chunk]:
    """Split on headings; split long sections on blank lines so no chunk is much over MAX_CHARS."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        heading = re.match(r"^#{1,4}\s+(.*)", line)
        if heading:
            sections.append((heading.group(1).strip(), []))
        else:
            sections[-1][1].append(line)
    chunks: list[Chunk] = []
    for title, lines in sections:
        paragraphs = [p.strip() for p in "\n".join(lines).split("\n\n") if p.strip()]
        buffer = ""
        pieces = []
        for paragraph in paragraphs:
            if buffer and len(buffer) + len(paragraph) > MAX_CHARS:
                pieces.append(buffer)
                buffer = ""
            buffer = f"{buffer}\n\n{paragraph}".strip()
        if buffer:
            pieces.append(buffer)
        for piece in pieces:
            body = f"{title}\n{piece}" if title else piece
            if len(body) >= MIN_CHARS:
                chunks.append(Chunk(f"{source}#{len(chunks) + 1}", source, title, body[: MAX_CHARS * 2], kind))
    return chunks


def collect(root: Path, processed_dir: Path | None = None, store=None, lessons: int = 200, experiments: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pattern in DOCS:
        for path in sorted(root.glob(pattern)):
            chunks += chunk_markdown(str(path.relative_to(root)), path.read_text(encoding="utf-8"), "doc")
    if processed_dir and processed_dir.exists():
        for name in REPORTS:
            path = processed_dir / name
            if path.exists():
                chunks += chunk_markdown(f"report:{name}", path.read_text(encoding="utf-8"), "report")
    if store is not None:
        for lesson in store.lessons(limit=lessons):
            chunks.append(Chunk(f"lesson:{lesson['id']}", f"lesson:{lesson['id']}", "lesson", lesson["text"], "lesson"))
        for entry in store.recent(None, limit=experiments):
            text = (f"Experiment ({entry['verdict']}): {entry['hypothesis']}. {entry['conclusion']} "
                    f"Config: {entry['name']}")
            chunks.append(Chunk(f"experiment:{entry['id']}", f"experiment:{entry['id']}", "experiment", text, "experiment"))
    return chunks
