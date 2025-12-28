from __future__ import annotations

import time
from typing import Any

from openai import AsyncOpenAI

from app.domains.models import DomainSpec
from app.schemas.run import PipelineResult
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


async def run_prompt_only(
    *,
    client: AsyncOpenAI,
    settings: Settings,
    domain: DomainSpec,
    query: str,
) -> PipelineResult:
    started = time.perf_counter()
    generation_config = _generation_config(settings)

    system_parts = [
        "You are a helpful assistant.",
        "If you do not know, say you don't know.",
    ]
    if domain.domain_prompt_prefix:
        system_parts.append(domain.domain_prompt_prefix.strip())

    resp = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": "\n".join(system_parts)},
            {"role": "user", "content": query},
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
        pipeline="prompt",
        model=settings.openai_model,
        generation_config=generation_config,
        answer=answer,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_estimate_usd=_estimate_cost_usd(settings, tokens_in, tokens_out),
        retrieved_chunks=None,
        flags={},
        error=None,
    )

