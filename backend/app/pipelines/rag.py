from __future__ import annotations

import re
import time
from typing import Any

from openai import AsyncOpenAI

from app.domains.models import DomainSpec
from app.pipelines.common import estimate_cost_usd, generation_config
from app.rag.retriever import get_retrieved_chunks
from app.rag.store import load_index_meta
from app.schemas.run import PipelineResult, RetrievedChunk
from app.settings import Settings


_CITATION_RE = re.compile(r"\[\d+\]")


def _format_context_with_budget(chunks: list[dict[str, Any]], max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    out_lines: list[str] = []
    used = 0
    for c in chunks:
        block = f"[{c.get('chunk_id')}] source: {c.get('source')}\n{c.get('text_preview')}\n"
        if used + len(block) > max_chars:
            break
        out_lines.append(block)
        used += len(block)
    return "\n".join(out_lines).strip()


def _is_unknown_answer(answer: str) -> bool:
    lowered = answer.lower()
    return "don't know" in lowered or "do not know" in lowered or "not in the context" in lowered


def _has_any_citation(answer: str) -> bool:
    return _CITATION_RE.search(answer) is not None


def _sum_optional_ints(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return a + b


async def run_rag(
    *,
    client: AsyncOpenAI,
    settings: Settings,
    domain: DomainSpec,
    query: str,
) -> PipelineResult:
    started = time.perf_counter()
    gen_config = generation_config(settings)

    if not domain.index_path.exists() or not domain.chunks_path.exists() or not domain.index_meta_path.exists():
        latency_ms = int((time.perf_counter() - started) * 1000)
        return PipelineResult(
            pipeline="rag",
            model=settings.openai_model,
            generation_config=gen_config,
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
            generation_config=gen_config,
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
            generation_config=gen_config,
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
    allowed_ids = sorted({int(c["chunk_id"]) for c in retrieved}) if retrieved else []
    allowed_ids_str = ", ".join(str(i) for i in allowed_ids)

    system_parts = [
        "You are a helpful assistant.",
        "Answer using ONLY the provided context.",
        "If the answer is not in the context, say you don't know.",
        "Do not guess or use outside knowledge. If the context does not explicitly support the claim, say you don't know.",
        "For questions about whether something is supported 'out of the box' or 'built-in', answer explicitly and state whether it's built-in or requires third-party integration (only if supported by the context).",
    ]
    if retrieved:
        system_parts.extend(
            [
                "Cite sources using bracketed chunk ids like [12].",
                "Every paragraph must include at least one citation.",
                f"You may ONLY cite these chunk ids: {allowed_ids_str}",
            ]
        )
    else:
        system_parts.append("No context was retrieved. Reply with 'I don't know.'")
    if domain.domain_prompt_prefix:
        system_parts.append(domain.domain_prompt_prefix.strip())

    resp = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": "\n".join(system_parts)},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"},
        ],
        temperature=gen_config["temperature"],
        top_p=gen_config["top_p"],
        max_tokens=gen_config["max_tokens"],
    )

    answer = resp.choices[0].message.content or ""
    tokens_in = getattr(resp.usage, "prompt_tokens", None)
    tokens_out = getattr(resp.usage, "completion_tokens", None)

    if retrieved and not _is_unknown_answer(answer) and not _has_any_citation(answer):
        flags["CITATION_REWRITE"] = True
        rewrite_system = [
            "You are a careful technical editor.",
            "Rewrite the draft answer so it includes citations using bracketed chunk ids like [12].",
            "Do not introduce any new facts that are not supported by the Context.",
            "If a claim cannot be supported by the Context, remove it or say you don't know.",
            "Every paragraph must include at least one citation.",
            f"You may ONLY cite these chunk ids: {allowed_ids_str}",
        ]
        rewrite = await client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "\n".join(rewrite_system)},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nDraft answer:\n{answer}\n\nRewrite with citations:",
                },
            ],
            temperature=gen_config["temperature"],
            top_p=gen_config["top_p"],
            max_tokens=gen_config["max_tokens"],
        )
        rewritten = rewrite.choices[0].message.content or ""
        if rewritten:
            answer = rewritten
        tokens_in = _sum_optional_ints(tokens_in, getattr(rewrite.usage, "prompt_tokens", None))
        tokens_out = _sum_optional_ints(tokens_out, getattr(rewrite.usage, "completion_tokens", None))
        if not _has_any_citation(answer):
            flags["MISSING_CITATIONS_AFTER_REWRITE"] = True

    latency_ms = int((time.perf_counter() - started) * 1000)

    return PipelineResult(
        pipeline="rag",
        model=settings.openai_model,
        generation_config=gen_config,
        answer=answer,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_estimate_usd=estimate_cost_usd(settings, model=settings.openai_model, tokens_in=tokens_in, tokens_out=tokens_out),
        retrieved_chunks=[RetrievedChunk(**c) for c in retrieved],
        flags=flags,
        error=None,
    )
