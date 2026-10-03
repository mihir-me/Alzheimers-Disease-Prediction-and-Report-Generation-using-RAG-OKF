"""Chunking and knowledge-base loading for the RAG pipeline."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import RagConfig


@dataclass
class Chunk:
    text: str
    source: str  # source document filename
    title: str   # nearest markdown heading


_SPLIT_RE = re.compile(r"(?m)^\s*$")


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _SPLIT_RE.split(text) if p.strip()]


def _nearest_heading(lines: list[str], idx: int) -> str:
    for line in reversed(lines[: idx + 1]):
        if line.startswith("# "):
            return line.lstrip("# ").strip()
    return "General"


def chunk_document(text: str, source: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Split a document into overlapping chunks at paragraph boundaries."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    paragraphs = _split_paragraphs(text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    chunk_no = 0

    def flush():
        nonlocal current, current_len, chunk_no
        if not current:
            return
        body = "\n\n".join(current)
        # find heading from the first paragraph's line index
        lines = text.splitlines()
        first = current[0]
        start = text.find(first)
        head = _nearest_heading(lines, text[:start].count("\n")) if start >= 0 else "General"
        chunk_no += 1
        chunks.append(Chunk(text=body, source=source, title=f"{head} (chunk {chunk_no})"))
        if overlap == 0:
            current = []
            current_len = 0
            return

        # keep trailing paragraphs for overlap
        keep: list[str] = []
        keep_len = 0
        for p in reversed(current):
            keep_len += len(p) + 2
            keep.insert(0, p)
            if keep_len >= overlap and len(keep) >= 2:
                break
        current = keep
        current_len = keep_len

    for para in paragraphs:
        para_len = len(para)
        if para_len > chunk_size:
            # over-long paragraph: flush current, then hard-split the paragraph
            flush()
            for i in range(0, len(para), chunk_size - overlap):
                pieces = para[i : i + chunk_size]
                if pieces.strip():
                    chunks.append(Chunk(text=pieces.strip(), source=source, title="General"))
            continue
        if current_len + para_len + 2 > chunk_size and current:
            flush()
        current.append(para)
        current_len += para_len + 2

    flush()
    # dedupe consecutive chunks with identical body
    seen: set[str] = set()
    out: list[Chunk] = []
    for c in chunks:
        if c.text in seen:
            continue
        seen.add(c.text)
        out.append(c)
    return out


def load_knowledge_base(cfg: RagConfig) -> list[Chunk]:
    """Load all markdown files in the knowledge directory and chunk them."""
    docs_dir: Path = cfg.knowledge_dir
    if not docs_dir.is_dir():
        return []
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_document(text, path.name, cfg.chunk_size, cfg.chunk_overlap))
    return chunks
