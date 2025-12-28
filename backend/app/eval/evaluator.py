from __future__ import annotations

import re
from typing import Any, Callable

from app.schemas.run import EvaluationResult, PipelineResult

RuleFn = Callable[[PipelineResult], tuple[list[str], list[str], str | None, float]]

_OVERCONFIDENT_RE = re.compile(r"\b(always|guaranteed|never|definitely|100%|must)\b", re.IGNORECASE)
_CITATION_RE = re.compile(r"\[(\d+)\]")


def overconfident_language_check(result: PipelineResult) -> tuple[list[str], list[str], str | None, float]:
    if _OVERCONFIDENT_RE.search(result.answer or "") is None:
        return [], [], None, 0.0
    return ["OVERCONFIDENT_LANGUAGE"], [], "Overconfident language detected.", 0.15


def not_grounded_check(result: PipelineResult) -> tuple[list[str], list[str], str | None, float]:
    if result.pipeline != "rag":
        return [], [], None, 0.0
    if not result.retrieved_chunks:
        return [], ["NO_EVIDENCE"], "No retrieved evidence available.", 0.25
    answer = result.answer or ""
    cited = {int(m.group(1)) for m in _CITATION_RE.finditer(answer)}
    if not cited:
        lowered = answer.lower()
        if "don't know" in lowered or "do not know" in lowered or "not in the context" in lowered:
            return [], [], None, 0.0
        return [], ["MISSING_CITATIONS"], "No chunk citations found (expected like [12]).", 0.25
    retrieved_ids = {c.chunk_id for c in result.retrieved_chunks}
    invalid = sorted(cited - retrieved_ids)
    if invalid:
        shown = ", ".join(str(i) for i in invalid[:5])
        more = "..." if len(invalid) > 5 else ""
        return (
            [],
            ["INVALID_CITATIONS"],
            f"Citations refer to chunks not in retrieved evidence: {shown}{more}",
            0.25,
        )
    return [], [], None, 0.0


RULES: dict[str, RuleFn] = {
    "overconfident_language_check": overconfident_language_check,
    "not_grounded_check": not_grounded_check,
}


def evaluate_results(
    *,
    results: dict[str, PipelineResult],
    rule_names: list[str],
) -> tuple[dict[str, EvaluationResult], dict[str, Any]]:
    evaluations: dict[str, EvaluationResult] = {}

    for pipeline, result in results.items():
        halluc_flags: list[str] = []
        grounding_flags: list[str] = []
        notes: list[str] = []
        penalty = 0.0

        for name in rule_names:
            rule = RULES.get(name)
            if rule is None:
                continue
            h, g, note, p = rule(result)
            halluc_flags.extend(h)
            grounding_flags.extend(g)
            if note:
                notes.append(note)
            penalty += p

        base = 1.0 if result.error is None else 0.0
        score = max(0.0, min(1.0, base - penalty))

        evaluations[pipeline] = EvaluationResult(
            quality_score=score if result.error is None else None,
            hallucination_flags=halluc_flags,
            grounding_flags=grounding_flags,
            notes=" ".join(notes) if notes else None,
        )

    summary = _summarize(results, evaluations)
    return evaluations, summary


def _summarize(
    results: dict[str, PipelineResult], evaluations: dict[str, EvaluationResult]
) -> dict[str, Any]:
    def safe_min_latency() -> tuple[str | None, int | None]:
        best: tuple[str | None, int | None] = (None, None)
        for k, v in results.items():
            if v.error is not None:
                continue
            if best[1] is None or v.latency_ms < best[1]:
                best = (k, v.latency_ms)
        return best

    def safe_min_cost() -> tuple[str | None, float | None]:
        best: tuple[str | None, float | None] = (None, None)
        for k, v in results.items():
            if v.error is not None or v.cost_estimate_usd is None:
                continue
            if best[1] is None or v.cost_estimate_usd < best[1]:
                best = (k, v.cost_estimate_usd)
        return best

    def safe_max_quality() -> tuple[str | None, float | None, list[str] | None]:
        scored: dict[str, float] = {
            k: float(v.quality_score) for k, v in evaluations.items() if v.quality_score is not None
        }
        if not scored:
            return None, None, None
        best_score = max(scored.values())
        winners = sorted([k for k, s in scored.items() if s == best_score])
        if len(winners) == 1:
            return winners[0], best_score, None
        return None, best_score, winners

    winner_by_latency, _ = safe_min_latency()
    winner_by_cost, _ = safe_min_cost()
    winner_by_quality, _, winner_by_quality_ties = safe_max_quality()

    summary: dict[str, Any] = {
        "winner_by_quality": winner_by_quality,
        "winner_by_quality_ties": winner_by_quality_ties,
        "winner_by_latency": winner_by_latency,
        "winner_by_cost": winner_by_cost,
        "tradeoff_summary": "Compare quality vs latency vs cost; prefer grounded answers when needed.",
    }
    if summary["winner_by_quality_ties"] is None:
        summary.pop("winner_by_quality_ties", None)
    return summary
