"""Embeddings: OpenAI-compatible API with a deterministic local fallback."""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from .config import RagConfig


class Embedder:
    """Embed text via the configured OpenAI-compatible endpoint.

    Falls back to a deterministic local hashing embedder when no API key is
    configured or when the API call fails, so the pipeline always works.
    """

    def __init__(self, cfg: RagConfig):
        self.cfg = cfg
        self._client: Any = None
        self._api_ok = False
        self._force_local = False

    def _get_client(self) -> Any:
        """Lazily create the OpenAI-compatible client (pickle-safe caching)."""
        if self._force_local:
            return None
        if self._client is None and self.cfg.has_api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.base_url)
            except Exception:
                self._client = None
        return self._client

    @property
    def provider(self) -> str:
        if self._force_local:
            return "local-hashing"
        return "openai" if self._api_ok else "local-hashing"

    def force_local(self):
        self._force_local = True
        self._api_ok = False

    def embed_batch(self, texts: list[str], dimension: int = 1024) -> np.ndarray:
        """Return an (n, dim) float32 matrix of L2-normalized embeddings."""
        if dimension < 1:
            raise ValueError("Embedding dimension must be positive")
        texts = [str(text) for text in texts]
        if not texts:
            return np.zeros((0, dimension), dtype=np.float32)
        client = self._get_client()
        if client is not None:
            try:
                resp = client.embeddings.create(
                    model=self.cfg.embedding_model,
                    input=[text[:8000] for text in texts],
                )
                rows = [d.embedding for d in resp.data]
                vectors = np.asarray(rows, dtype=np.float32)
                if vectors.ndim != 2 or vectors.shape[0] != len(texts):
                    raise ValueError("Embedding API returned an unexpected shape")
                if not np.isfinite(vectors).all():
                    raise ValueError("Embedding API returned non-finite values")
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                if np.any(norms == 0):
                    raise ValueError("Embedding API returned a zero vector")
                self._api_ok = True
                return (vectors / norms).astype(np.float32)
            except Exception:
                self._api_ok = False
        return self._embed_local(texts, dimension)

    def embed(self, text: str, dimension: int = 1024) -> np.ndarray:
        return self.embed_batch([text], dimension=dimension)[0]

    @staticmethod
    def _embed_local(texts: list[str], dimension: int = 1024) -> np.ndarray:
        """Deterministic char n-gram hashing embedder (offline fallback)."""
        if dimension < 1:
            raise ValueError("Embedding dimension must be positive")
        grams = [2, 3, 4]
        out = np.zeros((len(texts), dimension), dtype="float32")
        for i, t in enumerate(texts):
            norm = t.lower()
            ngram_set: set[str] = set()
            for n in grams:
                ngram_set.update(
                    norm[j : j + n] for j in range(len(norm) - n + 1)
                )
            if not ngram_set:
                ngram_set.add("")
            row = out[i]
            for g in ngram_set:
                h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
                idx = h % dimension
                sign = 1.0 if (h >> 31) & 1 else -1.0
                row[idx] += sign * math.sqrt(1.0 / max(len(ngram_set), 1))
            norm_l2 = float(np.linalg.norm(row))
            if norm_l2 > 0:
                row /= norm_l2
        return out
