from __future__ import annotations

from typing import Any

from app.settings import Settings


def system_prompt_parts(mode: str, domain_prompt_prefix: str | None) -> list[str]:
    if mode == "general":
        return [
            "You are a helpful assistant.",
            "Answer using your general knowledge.",
            "If you do not know, say you don't know.",
            "Do not guess or speculate about unknowns.",
        ]

    parts = [
        "You are a helpful assistant.",
        "If you do not know, say you don't know.",
        "Do not guess or speculate. If you are not sure, say you don't know.",
        "If a question asks you to answer only if documentation explicitly says so, only answer if you are certain; otherwise say you don't know.",
        "For questions about whether something is supported 'out of the box' or 'built-in', answer explicitly and state whether it's built-in or requires third-party integration.",
    ]
    if domain_prompt_prefix:
        parts.append(domain_prompt_prefix.strip())
    return parts


def generation_config(settings: Settings) -> dict[str, Any]:
    return {
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_output_tokens,
    }


def estimate_cost_usd(
    settings: Settings, *, model: str, tokens_in: int | None, tokens_out: int | None
) -> float | None:
    if tokens_in is None or tokens_out is None:
        return None
    if settings.openai_cost_input_per_1m is None or settings.openai_cost_output_per_1m is None:
        return None

    in_rate = float(settings.openai_cost_input_per_1m)
    out_rate = float(settings.openai_cost_output_per_1m)

    if model.startswith("ft:"):
        if (
            settings.openai_cost_input_per_1m_finetuned is not None
            and settings.openai_cost_output_per_1m_finetuned is not None
        ):
            in_rate = float(settings.openai_cost_input_per_1m_finetuned)
            out_rate = float(settings.openai_cost_output_per_1m_finetuned)
        else:
            in_rate *= 2.0
            out_rate *= 2.0

    return (tokens_in / 1_000_000) * in_rate + (tokens_out / 1_000_000) * out_rate
