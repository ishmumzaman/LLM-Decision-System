from __future__ import annotations

import time

from openai import AsyncOpenAI

from app.domains.models import DomainSpec
from app.pipelines.common import estimate_cost_usd, generation_config, system_prompt_parts
from app.schemas.run import PipelineResult
from app.settings import Settings


async def run_finetuned(
    *,
    client: AsyncOpenAI,
    settings: Settings,
    domain: DomainSpec,
    query: str,
) -> PipelineResult:
    started = time.perf_counter()
    gen_config = generation_config(settings)

    model = domain.finetuned_model or settings.openai_finetuned_model
    if not model:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return PipelineResult(
            pipeline="finetune",
            model=settings.openai_model,
            generation_config=gen_config,
            answer="",
            latency_ms=latency_ms,
            tokens_in=None,
            tokens_out=None,
            cost_estimate_usd=None,
            retrieved_chunks=None,
            flags={},
            error="MISSING_FINETUNED_MODEL: set OPENAI_FINETUNED_MODEL or domains/<domain>/config.yaml finetuned_model",
        )

    system_parts = system_prompt_parts(domain.domain_prompt_prefix)

    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "\n".join(system_parts)},
            {"role": "user", "content": query},
        ],
        temperature=gen_config["temperature"],
        top_p=gen_config["top_p"],
        max_tokens=gen_config["max_tokens"],
    )

    latency_ms = int((time.perf_counter() - started) * 1000)
    answer = resp.choices[0].message.content or ""
    tokens_in = getattr(resp.usage, "prompt_tokens", None)
    tokens_out = getattr(resp.usage, "completion_tokens", None)

    return PipelineResult(
        pipeline="finetune",
        model=model,
        generation_config=gen_config,
        answer=answer,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_estimate_usd=estimate_cost_usd(settings, model=model, tokens_in=tokens_in, tokens_out=tokens_out),
        retrieved_chunks=None,
        flags={},
        error=None,
    )
