"""High-level RAG report service used by the Streamlit app."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .chunking import load_knowledge_base
from .config import RagConfig, load_config
from .embeddings import Embedder
from .report_generator import generate_report
from .vector_store import Hit, VectorStore


class RagReportService:
    """Builds the FAISS index from the knowledge base and generates reports."""

    def __init__(self, cfg: RagConfig | None = None):
        self.cfg = cfg or load_config()
        self.embedder = Embedder(self.cfg)
        self.store = VectorStore(self.cfg, self.embedder)

    def ensure_index(self) -> int:
        """Build or load the FAISS index. Returns the number of chunks."""
        if self.store.n_total > 0:
            return self.store.n_total
        chunks = load_knowledge_base(self.cfg)
        if not chunks:
            return 0
        self.store.build(chunks)
        try:
            self.store.save()
        except Exception:
            pass
        return self.store.n_total

    def build_index(self) -> int:
        """Force a rebuild of the FAISS index from the knowledge base."""
        chunks = load_knowledge_base(self.cfg)
        if not chunks:
            return 0
        self.store.build(chunks)
        try:
            self.store.save()
        except Exception:
            pass
        return self.store.n_total

    def load_index(self) -> int:
        """Load a previously saved index. Returns chunk count or 0."""
        self.store = VectorStore.load(self.cfg, self.embedder)
        return self.store.n_total

    def retrieve(self, query: str, top_k: int | None = None) -> list[Hit]:
        try:
            self.ensure_index()
            return self.store.search(query, top_k)
        except (OSError, ValueError, RuntimeError, ImportError, KeyError, IndexError, TypeError):
            return []

    def generate(self, prediction: dict[str, Any], query: str) -> dict[str, Any]:
        """Retrieve context and generate the markdown report.

        Returns {"report": str, "used_llm": bool, "hits": list[Hit],
                 "index_size": int, "embedding_provider": str}
        """
        hits = self.retrieve(query)
        report, used_llm = generate_report(self.cfg, prediction, query, hits)
        return {
            "report": report,
            "used_llm": used_llm,
            "hits": hits,
            "index_size": self.store.n_total,
            "embedding_provider": self.embedder.provider,
        }
