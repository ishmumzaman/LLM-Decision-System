from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss  # type: ignore


@lru_cache(maxsize=8)
def load_index_meta(index_meta_path: str) -> dict[str, Any]:
    path = Path(index_meta_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing RAG index metadata file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("index_meta.json must be a JSON object")
    return data


@lru_cache(maxsize=8)
def load_faiss_index(index_path: str) -> faiss.Index:  # type: ignore[name-defined]
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing RAG index file: {path}")
    return faiss.read_index(str(path))


@lru_cache(maxsize=8)
def load_chunks(chunks_path: str) -> list[dict[str, Any]]:
    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing RAG chunks file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("chunks.json must be a JSON list")
    return data
