"""Configuration loading for the RAG pipeline (.env via python-dotenv)."""
from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _env_str(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_path(key: str, default: str) -> Path:
    value = Path(_env_str(key, default)).expanduser()
    return value if value.is_absolute() else _PROJECT_ROOT / value


def _env_int(key: str, default: int) -> int:
    value = _env_str(key, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _env_float(key: str, default: float) -> float:
    value = _env_str(key, str(default))
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be numeric") from exc


@dataclass
class RagConfig:
    api_key: str = field(default_factory=lambda: _env_str("OPENAI_API_KEY"))
    base_url: str = field(default_factory=lambda: _env_str("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    llm_model: str = field(default_factory=lambda: _env_str("LLM_MODEL", "gemini-2.0-flash"))
    embedding_model: str = field(default_factory=lambda: _env_str("EMBEDDING_MODEL", "text-embedding-3-small"))
    knowledge_dir: Path = field(
        default_factory=lambda: _env_path("KNOWLEDGE_DIR", "knowledge")
    )
    faiss_index_path: Path = field(
        default_factory=lambda: _env_path("FAISS_INDEX_PATH", "rag_pipeline/faiss_index")
    )
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 100))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    min_score: float = field(default_factory=lambda: _env_float("RAG_MIN_SCORE", 0.1))

    def __post_init__(self):
        self.chunk_size = max(1, self.chunk_size)
        self.chunk_overlap = max(0, min(self.chunk_overlap, self.chunk_size - 1))
        self.top_k = max(1, self.top_k)
        if not math.isfinite(self.min_score):
            raise ValueError("RAG_MIN_SCORE must be finite")
        self.min_score = min(1.0, max(0.0, self.min_score))

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key) and self.api_key.lower() not in {"", "your-api-key-here", "your_api_key_here"}


def load_config() -> RagConfig:
    return RagConfig()
