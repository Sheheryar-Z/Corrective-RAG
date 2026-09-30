"""Runtime configuration. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # Paths
    data_dir: Path = Path("data")
    index_dir: Path = Path(".index")

    # Models
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"

    # Chunking / retrieval
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 4

    # CRAG action thresholds (scores are in [0, 1])
    upper_threshold: float = 0.7
    lower_threshold: float = 0.3

    # Knowledge refinement
    sentences_per_strip: int = 1

    # Web search
    web_max_results: int = 5

    # Parallel LLM calls for per-document / per-strip scoring
    max_concurrency: int = 8

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            data_dir=Path(_env("CRAG_DATA_DIR", "data")),
            index_dir=Path(_env("CRAG_INDEX_DIR", ".index")),
            llm_model=_env("CRAG_LLM_MODEL", "gpt-4o-mini"),
            embedding_model=_env("CRAG_EMBEDDING_MODEL", "text-embedding-3-large"),
            chunk_size=int(_env("CRAG_CHUNK_SIZE", "900")),
            chunk_overlap=int(_env("CRAG_CHUNK_OVERLAP", "150")),
            top_k=int(_env("CRAG_TOP_K", "4")),
            upper_threshold=float(_env("CRAG_UPPER_THRESHOLD", "0.7")),
            lower_threshold=float(_env("CRAG_LOWER_THRESHOLD", "0.3")),
            sentences_per_strip=int(_env("CRAG_SENTENCES_PER_STRIP", "1")),
            web_max_results=int(_env("CRAG_WEB_MAX_RESULTS", "5")),
            max_concurrency=int(_env("CRAG_MAX_CONCURRENCY", "8")),
        )
