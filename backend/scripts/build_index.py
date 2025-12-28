from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss  # type: ignore
import numpy as np
from openai import OpenAI


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.domains.loader import load_domain_spec  # noqa: E402
from app.rag.chunking import chunk_text  # noqa: E402
from app.settings import Settings  # noqa: E402


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def _path_id(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _read_fetch_meta(documents_dir: Path) -> dict[str, str] | None:
    meta_path = documents_dir / "FETCH_META.txt"
    if not meta_path.exists():
        return None
    parsed: dict[str, str] = {}
    for line in meta_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k:
            parsed[k] = v
    return parsed or None


def _iter_doc_paths(*, documents_dir: Path, allowed_exts: list[str], max_docs: int | None) -> list[Path]:
    allowed = {f".{e.lower().lstrip('.')}" for e in allowed_exts}
    all_paths = sorted([p for p in documents_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed])
    return all_paths[:max_docs] if max_docs else all_paths


def _corpus_sha256(paths: list[Path]) -> str:
    hasher = hashlib.sha256()
    for p in paths:
        hasher.update(_path_id(p).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(p.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _build_chunks(*, doc_paths: list[Path], chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for p in doc_paths:
        text = p.read_text(encoding="utf-8", errors="ignore")
        rel = _path_id(p)
        for ch in chunk_text(text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            chunks.append({"source": rel, "text": ch.text, "start": ch.start, "end": ch.end})
    for i, c in enumerate(chunks):
        c["chunk_id"] = i
    return chunks


def _embed_texts(*, client: OpenAI, model: str, texts: list[str], batch_size: int) -> np.ndarray:
    if not texts:
        raise ValueError("No texts to embed")

    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)

        out: list[list[float] | None] = [None] * len(batch)
        for item in resp.data:
            out[item.index] = item.embedding
        vectors.extend([v for v in out if v is not None])

        print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks...")

    arr = np.asarray(vectors, dtype=np.float32)
    return _normalize(arr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build offline RAG artifacts (FAISS + chunks.json).")
    parser.add_argument("--domain", required=True, help="Domain name from domains/registry.yaml")
    parser.add_argument("--batch-size", type=int, default=96, help="Embedding batch size")
    args = parser.parse_args()

    settings = Settings()
    if settings.llm_provider != "openai":
        print(
            f"ERROR: unsupported LLM_PROVIDER={settings.llm_provider!r} (MVP supports only 'openai').",
            file=sys.stderr,
        )
        return 2
    if not settings.openai_api_key:
        print("ERROR: Missing OPENAI_API_KEY (set env var or backend/.env).", file=sys.stderr)
        return 2

    domain = load_domain_spec(settings.domains_dir, args.domain)
    documents_dir = domain.documents_dir
    if not documents_dir.exists():
        print(f"ERROR: documents dir not found: {documents_dir}", file=sys.stderr)
        return 2

    doc_paths = _iter_doc_paths(
        documents_dir=documents_dir, allowed_exts=domain.allowed_file_types, max_docs=domain.max_docs
    )
    if not doc_paths:
        print(f"ERROR: no docs found under {documents_dir} (allowed: {domain.allowed_file_types})", file=sys.stderr)
        return 2

    corpus_hash = _corpus_sha256(doc_paths)
    chunks = _build_chunks(doc_paths=doc_paths, chunk_size=domain.chunk_size, chunk_overlap=domain.chunk_overlap)
    if not chunks:
        print("ERROR: no chunks produced (check chunk_size / docs).", file=sys.stderr)
        return 2

    print(f"Building index for domain={domain.name} docs={len(doc_paths)} chunks={len(chunks)}")

    client = OpenAI(api_key=settings.openai_api_key)
    embeddings = _embed_texts(
        client=client,
        model=settings.openai_embedding_model,
        texts=[c["text"] for c in chunks],
        batch_size=max(1, args.batch_size),
    )

    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    domain.artifacts_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(domain.index_path))
    domain.chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")

    fetch_meta = _read_fetch_meta(documents_dir)

    meta = {
        "domain": domain.name,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": settings.openai_embedding_model,
        "chunk_size": domain.chunk_size,
        "chunk_overlap": domain.chunk_overlap,
        "doc_count": len(doc_paths),
        "chunk_count": len(chunks),
        "corpus_sha256": corpus_hash,
        "corpus_files": [_path_id(p) for p in doc_paths],
    }
    if fetch_meta:
        meta["source_repo"] = fetch_meta.get("source_repo")
        meta["source_ref"] = fetch_meta.get("ref") or None
        meta["source_commit"] = fetch_meta.get("commit") or None
        meta["source_path"] = fetch_meta.get("source_path") or None
    domain.index_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote: {domain.index_path}")
    print(f"Wrote: {domain.chunks_path}")
    print(f"Wrote: {domain.index_meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
