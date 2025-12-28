from __future__ import annotations

import time
from typing import Any

from openai import AsyncOpenAI

from app.domains.models import DomainSpec
from app.rag.retriever import get_retrieved_chunks
from app.rag.store import load_index_meta
from app.schemas.run import PipelineResult, RetrievedChunk
from app.settings import Settings


def _generation_config(settings: Settings) -> dict[str, Any]:
    return {
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_output_tokens,
    }


def _estimate_cost_usd(settings: Settings, tokens_in: int | None, tokens_out: int | None) -> float | None:
    if tokens_in is None or tokens_out is None:
        return None
    if settings.openai_cost_input_per_1m is None or settings.openai_cost_output_per_1m is None:
        return None
    return (tokens_in / 1_000_000) * settings.openai_cost_input_per_1m + (
        tokens_out / 1_000_000
    ) * settings.openai_cost_output_per_1m


def _format_context(chunks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for c in chunks:
        lines.append(f"[{c['chunk_id']}] source: {c['source']}")
        lines.append(c["text_preview"])
        lines.append("")
    return "\n".join(lines).strip()


def _format_context_with_budget(chunks: list[dict[str, Any]], max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    out_lines: list[str] = []
    used = 0
    for c in chunks:
        block = f"[{c['chunk_id']}] source: {c['source']}\n{c['text_preview']}\n"
        if used + len(block) > max_chars:
            break
        out_lines.append(block)
        used += len(block)
    return "\n".join(out_lines).strip()


async def run_rag(
    *,
    client: AsyncOpenAI,
    settings: Settings,
    domain: DomainSpec,
    query: str,
) -> PipelineResult:
    started = time.perf_counter()
    generation_config = _generation_config(settings)

    if not domain.index_path.exists() or not domain.chunks_path.exists() or not domain.index_meta_path.exists():
        latency_ms = int((time.perf_counter() - started) * 1000)
        return PipelineResult(
            pipeline="rag",
            model=settings.openai_model,
            generation_config=generation_config,
            answer="",
            latency_ms=latency_ms,
            tokens_in=None,
            tokens_out=None,
            cost_estimate_usd=None,
            retrieved_chunks=None,
            flags={},
            error=f"MISSING_RAG_ARTIFACTS: build with `python backend/scripts/build_index.py --domain {domain.name}`",
        )

    try:
        meta = load_index_meta(str(domain.index_meta_path))
    except Exception as exc:  # noqa: BLE001
        latency_ms = int((time.perf_counter() - started) * 1000)
        return PipelineResult(
            pipeline="rag",
            model=settings.openai_model,
            generation_config=generation_config,
            answer="",
            latency_ms=latency_ms,
            tokens_in=None,
            tokens_out=None,
            cost_estimate_usd=None,
            retrieved_chunks=None,
            flags={},
            error=f"INVALID_RAG_INDEX_META: {type(exc).__name__}",
        )

    if meta.get("embedding_model") and meta.get("embedding_model") != settings.openai_embedding_model:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return PipelineResult(
            pipeline="rag",
            model=settings.openai_model,
            generation_config=generation_config,
            answer="",
            latency_ms=latency_ms,
            tokens_in=None,
            tokens_out=None,
            cost_estimate_usd=None,
            retrieved_chunks=None,
            flags={
                "EMBEDDING_MODEL_MISMATCH": {
                    "expected": settings.openai_embedding_model,
                    "found": meta.get("embedding_model"),
                }
            },
            error="RAG_ARTIFACT_MISMATCH",
        )

    k = min(domain.retrieval_k, settings.rag_max_chunks) if settings.rag_max_chunks else domain.retrieval_k
    retrieved = await get_retrieved_chunks(
        client=client,
        domain=domain,
        embedding_model=settings.openai_embedding_model,
        query=query,
        k=k,
    )

    flags: dict[str, Any] = {}
    if not retrieved:
        flags["NO_RETRIEVAL_HITS"] = True

    context = _format_context_with_budget(retrieved, settings.rag_max_context_chars)

    system_parts = [
        "You are a helpful assistant.",
        "Answer using ONLY the provided context.",
        "If the answer is not in the context, say you don't know.",
        "Cite sources using bracketed chunk ids like [12].",
    ]
    if domain.domain_prompt_prefix:
        system_parts.append(domain.domain_prompt_prefix.strip())

    resp = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": "\n".join(system_parts)},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"},
        ],
        temperature=generation_config["temperature"],
        top_p=generation_config["top_p"],
        max_tokens=generation_config["max_tokens"],
    )

    latency_ms = int((time.perf_counter() - started) * 1000)
    answer = resp.choices[0].message.content or ""
    tokens_in = getattr(resp.usage, "prompt_tokens", None)
    tokens_out = getattr(resp.usage, "completion_tokens", None)

    return PipelineResult(
        pipeline="rag",
        model=settings.openai_model,
        generation_config=generation_config,
        answer=answer,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_estimate_usd=_estimate_cost_usd(settings, tokens_in, tokens_out),
        retrieved_chunks=[RetrievedChunk(**c) for c in retrieved],
        flags=flags,
        error=None,
    )
