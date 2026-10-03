"""FAISS vector store for the knowledge base (build + search + persist)."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .chunking import Chunk
from .config import RagConfig
from .embeddings import Embedder

INDEX_EXT = ".index"
META_EXT = ".pkl"


@dataclass
class Hit:
    text: str
    source: str
    title: str
    score: float


class VectorStore:
    """FAISS inner-product (cosine) index over knowledge chunks."""

    def __init__(self, cfg: RagConfig, embedder: Embedder):
        self.cfg = cfg
        self.embedder = embedder
        self.index: Any = None
        self.metadata: list[dict] = []
        self.embedding_provider = "local-hashing"
        self.embedding_dimension = 0

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            self.index = None
            self.metadata = []
            self.embedding_dimension = 0
            return
        texts = [c.text for c in chunks]
        vectors = self.embedder.embed_batch(texts)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] < 1:
            raise ValueError("Embedder returned an invalid vector matrix")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0) or not np.isfinite(vectors).all():
            raise ValueError("Embedder returned invalid vectors")
        vectors = (vectors / norms).astype(np.float32)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        self.metadata = [
            {"text": c.text, "source": c.source, "title": c.title} for c in chunks
        ]
        self.embedding_provider = self.embedder.provider
        self.embedding_dimension = dim

    def search(self, query: str, top_k: int | None = None) -> list[Hit]:
        if self.index is None or self.index.ntotal == 0:
            return []
        requested_k = self.cfg.top_k if top_k is None else top_k
        if requested_k < 1:
            return []
        if self.embedding_provider == "local-hashing":
            self.embedder.force_local()
        dimension = int(self.index.d)
        q = self.embedder.embed(query, dimension=dimension)
        if q.shape != (dimension,) or not np.isfinite(q).all():
            return []
        k = min(requested_k, self.index.ntotal, len(self.metadata))
        if k < 1:
            return []
        scores, idxs = self.index.search(q.reshape(1, -1).astype("float32"), k)
        hits: list[Hit] = []
        for score, j in zip(scores[0], idxs[0]):
            if j < 0 or float(score) < self.cfg.min_score:
                continue
            m = self.metadata[j]
            hits.append(Hit(text=m["text"], source=m["source"], title=m["title"], score=float(score)))
        return hits

    def save(self, path: Path | None = None) -> Path:
        if self.index is None:
            raise ValueError("Cannot save an empty vector index")
        path = Path(path or self.cfg.faiss_index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path) + INDEX_EXT)
        with open(str(path) + META_EXT, "wb") as f:
            pickle.dump(
                {
                    "metadata": self.metadata,
                    "provider": self.embedding_provider,
                    "dimension": self.embedding_dimension,
                    "embedding_model": self.cfg.embedding_model,
                    "min_score": self.cfg.min_score,
                },
                f,
            )
        return path

    @classmethod
    def load(cls, cfg: RagConfig, embedder: Embedder) -> "VectorStore":
        path = Path(cfg.faiss_index_path)
        vs = cls(cfg, embedder)
        idx_file = str(path) + INDEX_EXT
        meta_file = str(path) + META_EXT
        if Path(idx_file).is_file() and Path(meta_file).is_file():
            try:
                index = faiss.read_index(idx_file)
                with open(meta_file, "rb") as f:
                    data = pickle.load(f)
                metadata = data.get("metadata", [])
                if (
                    index.d > 0
                    and index.ntotal == len(metadata)
                    and index.d == int(data.get("dimension", index.d))
                ):
                    vs.index = index
                    vs.metadata = metadata
                    vs.embedding_provider = data.get("provider", "local-hashing")
                    vs.embedding_dimension = index.d
                    if vs.embedding_provider == "local-hashing":
                        embedder.force_local()
            except Exception:
                vs.index = None
                vs.metadata = []
        return vs

    @property
    def n_total(self) -> int:
        return self.index.ntotal if self.index is not None else 0
