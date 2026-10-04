"""Chunking and knowledge-base loading for the RAG pipeline."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import RagConfig
from .okf_loader import Concept, OkfBundle, load_bundle


@dataclass
class Chunk:
    text: str
    source: str  # source document filename
    title: str   # nearest markdown heading
    concept_id: str = ""  # OKF concept id, e.g. "scores/cdr_scale"
    concept_type: str = ""  # OKF concept type, e.g. "Clinical Score"
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


_SPLIT_RE = re.compile(r"(?m)^\s*$")


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _SPLIT_RE.split(text) if p.strip()]


def _nearest_heading(lines: list[str], idx: int) -> str:
    for line in reversed(lines[: idx + 1]):
        if line.startswith("# "):
            return line.lstrip("# ").strip()
    return "General"


def chunk_document(
    text: str,
    source: str,
    chunk_size: int,
    overlap: int,
    *,
    concept_id: str = "",
    concept_type: str = "",
    default_title: str = "General",
) -> list[Chunk]:
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

    def add(body: str, heading: str) -> None:
        nonlocal chunk_no
        chunk_no += 1
        chunks.append(
            Chunk(
                text=body,
                source=source,
                title=f"{heading} (chunk {chunk_no})",
                concept_id=concept_id,
                concept_type=concept_type,
                chunk_index=chunk_no,
            )
        )

    def flush():
        nonlocal current, current_len
        if not current:
            return
        body = "\n\n".join(current)
        # find heading from the first paragraph's line index
        lines = text.splitlines()
        first = current[0]
        start = text.find(first)
        head = (
            _nearest_heading(lines, text[:start].count("\n"))
            if start >= 0
            else default_title
        )
        add(body, head)
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
                    add(pieces.strip(), default_title)
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


def chunk_concept(
    concept: Concept,
    chunk_size: int,
    overlap: int,
    *,
    prepend_title: bool = True,
) -> list[Chunk]:
    """Chunk one OKF concept, keeping its metadata on every chunk.

    The concept title is prepended to the chunk text so the embedding sees the
    topic it belongs to. Chunk sizes are unchanged: a concept that fits inside
    ``chunk_size`` stays a single chunk, longer concepts are split.
    """
    header = f"{concept.title}\n\n" if prepend_title else ""
    header_len = len(header)
    budget = max(1, chunk_size - header_len)
    body = concept.body
    if not body.strip():
        return [
            Chunk(
                text=f"{header}{concept.description}".strip(),
                source=concept.source,
                title=concept.title,
                concept_id=concept.concept_id,
                concept_type=concept.type,
                chunk_index=1,
            )
        ]
    parts = chunk_document(
        body,
        concept.source,
        budget,
        min(overlap, budget - 1),
        concept_id=concept.concept_id,
        concept_type=concept.type,
        default_title=concept.title,
    )
    for part in parts:
        part.text = f"{header}{part.text}".strip()
        part.title = concept.title
    return parts


def build_concept_chunks(
    concepts: list[Concept] | OkfBundle,
    cfg: RagConfig,
) -> list[Chunk]:
    """Chunk a bundle (or an explicit concept list) for indexing."""
    items = list(concepts) if isinstance(concepts, OkfBundle) else list(concepts)
    chunks: list[Chunk] = []
    for concept in items:
        chunks.extend(chunk_concept(concept, cfg.chunk_size, cfg.chunk_overlap))
    return chunks


def load_bundle_chunks(cfg: RagConfig, bundle: OkfBundle | None = None) -> list[Chunk]:
    """Load the OKF bundle and chunk it concept by concept."""
    bundle = bundle if bundle is not None else load_bundle(cfg.knowledge_bundle_dir)
    return build_concept_chunks(bundle, cfg)


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
