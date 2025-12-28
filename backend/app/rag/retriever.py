from __future__ import annotations

import numpy as np
from openai import AsyncOpenAI

from app.domains.models import DomainSpec
from app.rag.store import load_chunks, load_faiss_index


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


async def embed_query(client: AsyncOpenAI, model: str, query: str) -> np.ndarray:
    resp = await client.embeddings.create(model=model, input=[query])
    vec = np.asarray(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    return _normalize(vec)


async def retrieve_top_k(
    *,
    client: AsyncOpenAI,
    domain: DomainSpec,
    embedding_model: str,
    query: str,
    k: int,
) -> list[tuple[int, float]]:
    index = load_faiss_index(str(domain.index_path))
    q = await embed_query(client, embedding_model, query)

    scores, ids = index.search(q, k)
    results: list[tuple[int, float]] = []
    for chunk_id, score in zip(ids[0].tolist(), scores[0].tolist(), strict=False):
        if chunk_id < 0:
            continue
        results.append((int(chunk_id), float(score)))
    return results


async def get_retrieved_chunks(
    *,
    client: AsyncOpenAI,
    domain: DomainSpec,
    embedding_model: str,
    query: str,
    k: int,
    preview_chars: int = 1200,
) -> list[dict]:
    chunks = load_chunks(str(domain.chunks_path))
    hits = await retrieve_top_k(
        client=client,
        domain=domain,
        embedding_model=embedding_model,
        query=query,
        k=k,
    )

    out: list[dict] = []
    for chunk_id, score in hits:
        if chunk_id >= len(chunks):
            continue
        c = chunks[chunk_id]
        text = str(c.get("text", ""))
        out.append(
            {
                "chunk_id": chunk_id,
                "source": str(c.get("source", "")),
                "text_preview": text[:preview_chars],
                "score": score,
            }
        )
    return out

