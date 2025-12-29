from __future__ import annotations

import json
import time
from typing import Any

from openai import AsyncOpenAI

from app.pipelines.common import estimate_cost_usd
from app.schemas.run import (
    AnswerabilityProxy,
    EvidenceSupportProxy,
    PipelineResult,
    ProxiesResult,
    ProxyAnswerabilityLabel,
    RunMode,
)
from app.settings import Settings


def _truncate(text: str, max_chars: int) -> str:
    s = (text or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 1)] + "ƒ?İ"


def _try_parse_json(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None


def _chunks_block(*, mode: RunMode, rag: PipelineResult | None, max_chunk_chars: int) -> str:
    if mode != "docs" or rag is None or not isinstance(rag.retrieved_chunks, list):
        return ""
    if not rag.retrieved_chunks:
        return ""
    parts: list[str] = []
    for c in rag.retrieved_chunks:
        parts.append(f"- [{c.chunk_id}] {c.source}\n{_truncate(c.text_preview or '', max_chunk_chars)}")
    return "\n\nContext snippets (previews):\n" + "\n\n".join(parts)


async def run_proxies(
    *,
    client: AsyncOpenAI,
    settings: Settings,
    mode: RunMode,
    domain: str,
    query: str,
    results: dict[str, PipelineResult],
    judge_model: str | None,
    judge_answer_chars: int,
    judge_chunk_chars: int,
    proxy_evidence: bool,
    proxy_answerability: bool,
) -> ProxiesResult:
    model = (judge_model or settings.openai_judge_model or settings.openai_model).strip()
    if not model:
        return ProxiesResult(error="MISSING_JUDGE_MODEL")

    rag = results.get("rag")
    chunks_block = _chunks_block(mode=mode, rag=rag, max_chunk_chars=judge_chunk_chars)

    total_latency_ms = 0
    total_tokens_in: int | None = 0
    total_tokens_out: int | None = 0
    total_cost: float | None = 0.0

    out = ProxiesResult(model=model)

    if proxy_answerability:
        sys = (
            "You are labeling whether a user question can be answered correctly without consulting the documentation.\n"
            "Return ONLY JSON with keys: label, answerable_without_docs, confidence, rationale.\n"
            "label must be one of: general, requires_docs, unsupported, unknown.\n"
            "\n"
            "Definitions:\n"
            "- general: a capable engineer could answer from general knowledge\n"
            "- requires_docs: needs doc-specific details/wording/version-specific behavior\n"
            "- unsupported: the question asks for something that likely isn't in the docs OR asks to answer only if docs say so\n"
            "- unknown: you are not sure\n"
            "\n"
            "Rules:\n"
            "- Do not reward verbosity or confident tone.\n"
            "- Use temperature-0 style: be conservative.\n"
            "- confidence is 0..1.\n"
            "No markdown, no extra keys."
        )
        user = f"domain: {domain}\nquestion: {query}\nReturn the JSON now."
        started = time.perf_counter()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                temperature=0,
                max_tokens=220,
            )
            total_latency_ms += int((time.perf_counter() - started) * 1000)

            tokens_in = getattr(resp.usage, "prompt_tokens", None)
            tokens_out = getattr(resp.usage, "completion_tokens", None)
            if isinstance(tokens_in, int) and total_tokens_in is not None:
                total_tokens_in += tokens_in
            else:
                total_tokens_in = None
            if isinstance(tokens_out, int) and total_tokens_out is not None:
                total_tokens_out += tokens_out
            else:
                total_tokens_out = None

            cost_est = estimate_cost_usd(
                settings,
                model=model,
                tokens_in=tokens_in if isinstance(tokens_in, int) else None,
                tokens_out=tokens_out if isinstance(tokens_out, int) else None,
            )
            if isinstance(cost_est, (int, float)) and total_cost is not None:
                total_cost += float(cost_est)
            else:
                total_cost = None

            parsed = _try_parse_json(resp.choices[0].message.content or "")
            if not isinstance(parsed, dict):
                out.answerability = AnswerabilityProxy(error="INVALID_JSON")
            else:
                label_raw = parsed.get("label")
                label: ProxyAnswerabilityLabel | None = None
                if isinstance(label_raw, str) and label_raw in {"general", "requires_docs", "unsupported", "unknown"}:
                    label = label_raw  # type: ignore[assignment]
                answerable = parsed.get("answerable_without_docs")
                if not isinstance(answerable, bool):
                    answerable = None
                confidence = parsed.get("confidence")
                if not isinstance(confidence, (int, float)):
                    confidence = None
                rationale = parsed.get("rationale")
                if not isinstance(rationale, str):
                    rationale = None
                out.answerability = AnswerabilityProxy(
                    label=label,
                    answerable_without_docs=answerable,
                    confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
                    rationale=rationale,
                )
        except Exception as exc:  # noqa: BLE001
            out.answerability = AnswerabilityProxy(error=f"{type(exc).__name__}: {exc}")

    if proxy_evidence:
        if mode != "docs":
            out.evidence_support = {}
        elif not chunks_block:
            for p, r in results.items():
                if r.error is not None:
                    continue
                out.evidence_support[p] = EvidenceSupportProxy(error="NO_RAG_CONTEXT_AVAILABLE")
        else:
            sys = (
                "You are verifying whether an answer's factual claims are supported by the provided documentation context snippets.\n"
                "Treat the context snippets as the ONLY allowed source. Do not use outside knowledge.\n"
                "\n"
                "Return ONLY JSON with keys: support_score, unsupported_claims, rationale.\n"
                "- support_score must be 0, 1, or 2.\n"
                "  0 = mostly unsupported (claims not in context)\n"
                "  1 = partially supported (mix of supported and unsupported)\n"
                "  2 = fully supported OR correctly abstains when context is insufficient\n"
                "- unsupported_claims: list up to 5 short statements from the answer that are not supported by the context.\n"
                "- rationale: one short sentence.\n"
                "\n"
                "Important:\n"
                "- Do NOT reward verbosity.\n"
                "- If the answer says \"I don't know\" and the context does not contain the requested info, that is GOOD (support_score=2).\n"
                "No markdown, no extra keys."
            )

            for p, r in results.items():
                if r.error is not None:
                    continue
                answer = _truncate(r.answer or "", judge_answer_chars)
                user = (
                    f"mode: {mode}\n"
                    f"domain: {domain}\n"
                    f"question: {query}\n\n"
                    f"answer:\n{answer}"
                    f"{chunks_block}\n\n"
                    "Return the JSON now."
                )
                started = time.perf_counter()
                try:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                        temperature=0,
                        max_tokens=320,
                    )
                    total_latency_ms += int((time.perf_counter() - started) * 1000)

                    tokens_in = getattr(resp.usage, "prompt_tokens", None)
                    tokens_out = getattr(resp.usage, "completion_tokens", None)
                    if isinstance(tokens_in, int) and total_tokens_in is not None:
                        total_tokens_in += tokens_in
                    else:
                        total_tokens_in = None
                    if isinstance(tokens_out, int) and total_tokens_out is not None:
                        total_tokens_out += tokens_out
                    else:
                        total_tokens_out = None

                    cost_est = estimate_cost_usd(
                        settings,
                        model=model,
                        tokens_in=tokens_in if isinstance(tokens_in, int) else None,
                        tokens_out=tokens_out if isinstance(tokens_out, int) else None,
                    )
                    if isinstance(cost_est, (int, float)) and total_cost is not None:
                        total_cost += float(cost_est)
                    else:
                        total_cost = None

                    parsed = _try_parse_json(resp.choices[0].message.content or "")
                    if not isinstance(parsed, dict):
                        out.evidence_support[p] = EvidenceSupportProxy(error="INVALID_JSON")
                        continue
                    support_score = parsed.get("support_score")
                    if not isinstance(support_score, (int, float)):
                        support_score = None
                    else:
                        support_score = float(max(0.0, min(2.0, float(support_score))))
                    unsupported = parsed.get("unsupported_claims")
                    unsupported_claims: list[str] = []
                    if isinstance(unsupported, list):
                        unsupported_claims = [str(x) for x in unsupported if x is not None][:5]
                    rationale = parsed.get("rationale")
                    if not isinstance(rationale, str):
                        rationale = None
                    out.evidence_support[p] = EvidenceSupportProxy(
                        support_score=support_score if isinstance(support_score, (int, float)) else None,
                        unsupported_claims=unsupported_claims,
                        rationale=rationale,
                    )
                except Exception as exc:  # noqa: BLE001
                    out.evidence_support[p] = EvidenceSupportProxy(error=f"{type(exc).__name__}: {exc}")

    out.latency_ms = int(total_latency_ms) if total_latency_ms else 0
    out.tokens_in = total_tokens_in
    out.tokens_out = total_tokens_out
    out.cost_estimate_usd = total_cost
    return out
