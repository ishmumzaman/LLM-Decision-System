from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from openai import AsyncOpenAI

from app.pipelines.common import estimate_cost_usd
from app.schemas.run import JudgeResult, PipelineResult, RunMode
from app.settings import Settings


_CRITERIA = ("correctness", "groundedness", "hallucination", "abstention", "usefulness", "conciseness")
_WEIGHTS: dict[str, float] = {
    "correctness": 3.0,
    "groundedness": 3.0,
    "hallucination": 2.0,
    "abstention": 2.0,
    "usefulness": 2.0,
    "conciseness": 1.0,
}
_MAX_POINTS = sum(_WEIGHTS.values()) * 2.0


def _truncate(text: str, max_chars: int) -> str:
    s = (text or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max(0, max_chars - 1)] + "…"


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


def _stable_pair_mapping(*, domain: str, mode: RunMode, query: str, p1: str, p2: str) -> tuple[str, str]:
    key = f"{domain}|{mode}|{query}|{p1}|{p2}".encode("utf-8", errors="ignore")
    b = hashlib.sha256(key).digest()[0]
    if b % 2 == 0:
        return p1, p2
    return p2, p1


def _clamp_0_2(v: object) -> float:
    if not isinstance(v, (int, float)):
        return 0.0
    return max(0.0, min(2.0, float(v)))


def _score_from_criteria(criteria: dict[str, object]) -> tuple[float, dict[str, float]]:
    normalized: dict[str, float] = {k: _clamp_0_2(criteria.get(k)) for k in _CRITERIA}
    points = sum(_WEIGHTS[k] * normalized[k] for k in _CRITERIA)
    score_0_10 = (points / _MAX_POINTS) * 10.0 if _MAX_POINTS > 0 else 0.0
    return float(score_0_10), normalized


def _build_prompt(
    *,
    mode: RunMode,
    domain: str,
    query: str,
    answer_a: str,
    answer_b: str,
    context_chunks: list[dict[str, Any]] | None,
    max_answer_chars: int,
    max_chunk_chars: int,
) -> tuple[str, str]:
    system = (
        "You are a strict evaluator for an LLM evaluation harness.\n"
        "You will be given a user question and two candidate answers (A and B).\n"
        "The answers are anonymized; do NOT try to guess which system produced them.\n"
        "\n"
        "IMPORTANT anti-bias rules:\n"
        "- Do NOT reward verbosity, formatting, or confident tone.\n"
        "- Prefer the shorter answer if both are equally correct and useful.\n"
        "- Penalize unnecessary length, filler, and overconfident claims.\n"
        "\n"
        "Scoring rubric per answer: give each criterion an integer 0, 1, or 2.\n"
        "- correctness: correct facts, answers the question\n"
        "- groundedness: in docs mode, claims should be supported by provided context OR clearly marked as uncertain; "
        "in general mode, avoid unsupported/confident speculation\n"
        "- hallucination: does not invent facts/APIs/steps\n"
        "- abstention: correctly says 'I don't know' / refuses when context is insufficient (docs mode), and does not make up details\n"
        "- usefulness: includes key points and minimal example when asked\n"
        "- conciseness: no fluff; avoids repetition; not overly long\n"
        "\n"
        "Mode rules:\n"
        "- mode='docs': Treat the provided context as the only authoritative source. If the context doesn't support the answer, "
        "correct abstention is preferred.\n"
        "- mode='general': You may use general knowledge; do NOT require citations.\n"
        "\n"
        "Return ONLY JSON with keys:\n"
        "{\n"
        "  \"criteria\": {\n"
        "    \"A\": {correctness:0|1|2, groundedness:0|1|2, hallucination:0|1|2, abstention:0|1|2, usefulness:0|1|2, conciseness:0|1|2},\n"
        "    \"B\": {same keys}\n"
        "  },\n"
        "  \"rationale\": \"short explanation\"\n"
        "}\n"
        "No markdown, no extra keys."
    )

    chunks_block = ""
    if mode == "docs" and context_chunks:
        previewed: list[str] = []
        for c in context_chunks[:3]:
            cid = c.get("chunk_id")
            src = c.get("source")
            txt = _truncate(str(c.get("text_preview") or ""), max_chunk_chars)
            previewed.append(f"- [{cid}] {src}\n{txt}")
        chunks_block = "\n\nContext snippets (previews):\n" + "\n\n".join(previewed)

    user = (
        f"mode: {mode}\n"
        f"domain: {domain}\n"
        f"question: {query}\n\n"
        f"=== Answer A ===\n{_truncate(answer_a, max_answer_chars)}\n\n"
        f"=== Answer B ===\n{_truncate(answer_b, max_answer_chars)}"
        f"{chunks_block}\n\n"
        "Return the JSON now."
    )

    return system, user


async def judge_run(
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
) -> JudgeResult:
    model = (judge_model or settings.openai_judge_model or settings.openai_model).strip()
    if not model:
        return JudgeResult(error="MISSING_JUDGE_MODEL")

    ok_pipelines = [p for p, r in results.items() if r.error is None]
    if len(ok_pipelines) < 2:
        return JudgeResult(model=model, error="NEED_AT_LEAST_TWO_PIPELINES")

    totals: dict[str, list[float]] = {p: [] for p in ok_pipelines}
    crit_accum: dict[str, dict[str, list[float]]] = {p: {k: [] for k in _CRITERIA} for p in ok_pipelines}

    judge_latency_total = 0
    judge_tokens_in_total: int | None = 0
    judge_tokens_out_total: int | None = 0
    judge_cost_total: float | None = 0.0

    rationale_lines: list[str] = []

    for i in range(len(ok_pipelines)):
        for j in range(i + 1, len(ok_pipelines)):
            p1 = ok_pipelines[i]
            p2 = ok_pipelines[j]
            r1 = results[p1]
            r2 = results[p2]

            if r1.error is not None and r2.error is not None:
                continue
            if r1.error is not None:
                totals[p2].append(10.0)
                totals[p1].append(0.0)
                rationale_lines.append(f"{p2} > {p1} (because {p1} errored)")
                continue
            if r2.error is not None:
                totals[p1].append(10.0)
                totals[p2].append(0.0)
                rationale_lines.append(f"{p1} > {p2} (because {p2} errored)")
                continue

            a_pipeline, b_pipeline = _stable_pair_mapping(domain=domain, mode=mode, query=query, p1=p1, p2=p2)
            a_text = results[a_pipeline].answer or ""
            b_text = results[b_pipeline].answer or ""

            # In docs mode, only include context when comparing against RAG,
            # using the retrieved chunks that RAG actually saw.
            context_chunks = None
            if mode == "docs" and "rag" in {p1, p2}:
                rag = results.get("rag")
                if rag and isinstance(rag.retrieved_chunks, list):
                    context_chunks = [c.model_dump() for c in rag.retrieved_chunks]

            system, user = _build_prompt(
                mode=mode,
                domain=domain,
                query=query,
                answer_a=a_text,
                answer_b=b_text,
                context_chunks=context_chunks,
                max_answer_chars=judge_answer_chars,
                max_chunk_chars=judge_chunk_chars,
            )

            started = time.perf_counter()
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0,
                    max_tokens=450,
                )
            except Exception as exc:  # noqa: BLE001
                return JudgeResult(model=model, error=f"{type(exc).__name__}: {exc}")

            latency_ms = int((time.perf_counter() - started) * 1000)
            judge_latency_total += latency_ms

            tokens_in = getattr(resp.usage, "prompt_tokens", None)
            tokens_out = getattr(resp.usage, "completion_tokens", None)
            if isinstance(tokens_in, int) and judge_tokens_in_total is not None:
                judge_tokens_in_total += tokens_in
            else:
                judge_tokens_in_total = None
            if isinstance(tokens_out, int) and judge_tokens_out_total is not None:
                judge_tokens_out_total += tokens_out
            else:
                judge_tokens_out_total = None

            cost_estimate = estimate_cost_usd(
                settings,
                model=model,
                tokens_in=tokens_in if isinstance(tokens_in, int) else None,
                tokens_out=tokens_out if isinstance(tokens_out, int) else None,
            )
            if isinstance(cost_estimate, (int, float)) and judge_cost_total is not None:
                judge_cost_total += float(cost_estimate)
            else:
                judge_cost_total = None

            content = resp.choices[0].message.content or ""
            parsed = _try_parse_json(content)
            if not isinstance(parsed, dict):
                return JudgeResult(model=model, error="INVALID_JSON")

            criteria = parsed.get("criteria")
            if not isinstance(criteria, dict):
                return JudgeResult(model=model, error="MISSING_CRITERIA")
            a_crit_raw = criteria.get("A")
            b_crit_raw = criteria.get("B")
            if not isinstance(a_crit_raw, dict) or not isinstance(b_crit_raw, dict):
                return JudgeResult(model=model, error="INVALID_CRITERIA")

            score_a, a_crit = _score_from_criteria(a_crit_raw)
            score_b, b_crit = _score_from_criteria(b_crit_raw)

            totals[a_pipeline].append(score_a)
            totals[b_pipeline].append(score_b)
            for k in _CRITERIA:
                crit_accum[a_pipeline][k].append(a_crit[k])
                crit_accum[b_pipeline][k].append(b_crit[k])

            winner = "tie"
            if abs(score_a - score_b) > 0.5:
                winner = a_pipeline if score_a > score_b else b_pipeline

            rationale_raw = parsed.get("rationale")
            rationale_s = str(rationale_raw) if isinstance(rationale_raw, str) else ""
            rationale_lines.append(
                f"{winner}: {a_pipeline}={score_a:.1f} vs {b_pipeline}={score_b:.1f} ({_truncate(rationale_s, 140)})"
            )

    avg_scores: dict[str, float] = {}
    avg_criteria: dict[str, dict[str, float]] = {}
    for p in ok_pipelines:
        xs = totals.get(p) or []
        avg_scores[p] = float(sum(xs) / len(xs)) if xs else 0.0
        avg_criteria[p] = {}
        for k in _CRITERIA:
            ys = crit_accum[p][k]
            avg_criteria[p][k] = float(sum(ys) / len(ys)) if ys else 0.0

    if not avg_scores:
        return JudgeResult(model=model, error="NO_SCORES")

    sorted_items = sorted(avg_scores.items(), key=lambda kv: kv[1], reverse=True)
    if len(sorted_items) == 1:
        overall_winner = sorted_items[0][0]
    else:
        top1, top2 = sorted_items[0], sorted_items[1]
        overall_winner = "tie" if abs(top1[1] - top2[1]) <= 0.5 else top1[0]

    rationale = _truncate("\n".join(rationale_lines), 900) if rationale_lines else None

    return JudgeResult(
        winner=overall_winner,
        scores=avg_scores,
        criteria=avg_criteria,
        rationale=rationale,
        model=model,
        latency_ms=judge_latency_total,
        tokens_in=judge_tokens_in_total,
        tokens_out=judge_tokens_out_total,
        cost_estimate_usd=judge_cost_total,
        error=None,
    )

