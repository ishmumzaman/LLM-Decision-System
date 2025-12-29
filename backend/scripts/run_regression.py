from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class QueryCase:
    id: str
    query: str
    tags: list[str]
    expect: dict[str, Any]


_IDK_RE = re.compile(r"\b(i don't know|i do not know|not in (the )?context|unknown)\b", re.IGNORECASE)


def _contains_pattern(text: str, pattern: str) -> bool:
    p = str(pattern)
    if p.startswith("re:"):
        return re.search(p[3:], text, flags=re.IGNORECASE) is not None
    return p.lower() in text.lower()


def _check_expectations(answer: str, expect: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    if not expect:
        return 1.0, {}

    must_include = expect.get("must_include") or []
    must_not_include = expect.get("must_not_include") or []
    must_include_any = expect.get("must_include_any") or []
    expect_idk = bool(expect.get("expect_idk") or False)

    if not isinstance(must_include, list):
        must_include = [must_include]
    if not isinstance(must_not_include, list):
        must_not_include = [must_not_include]
    if not isinstance(must_include_any, list):
        must_include_any = [must_include_any]

    missing: list[str] = []
    forbidden: list[str] = []
    total = 0
    passed = 0

    for p in must_include:
        total += 1
        if isinstance(p, str) and _contains_pattern(answer, p):
            passed += 1
        else:
            missing.append(str(p))

    for group in must_include_any:
        total += 1
        if isinstance(group, str):
            group = [group]
        if not isinstance(group, list):
            missing.append(str(group))
            continue
        if any(isinstance(p, str) and _contains_pattern(answer, p) for p in group):
            passed += 1
        else:
            missing.append(" | ".join(str(p) for p in group))

    for p in must_not_include:
        total += 1
        if isinstance(p, str) and _contains_pattern(answer, p):
            forbidden.append(p)
        else:
            passed += 1

    if expect_idk:
        total += 1
        if _IDK_RE.search(answer or "") is not None:
            passed += 1
        else:
            missing.append("IDK_EXPECTED")

    score = (passed / total) if total > 0 else 1.0
    details: dict[str, Any] = {}
    if missing:
        details["missing"] = missing
    if forbidden:
        details["forbidden"] = forbidden
    return score, details

def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _http_json(
    *,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout_s: float = 60.0,
) -> tuple[int, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return int(resp.status), json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            body = raw
        return int(exc.code), body
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed: {url} ({exc})") from exc


def _load_suite(path: Path) -> tuple[str, str | None, list[QueryCase]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))  # type: ignore[no-untyped-call]

    suite_name = path.stem
    domain: str | None = None
    queries: list[Any]

    if isinstance(raw, dict):
        suite_name = str(raw.get("suite") or suite_name)
        domain = raw.get("domain")
        queries = raw.get("queries") or []
    elif isinstance(raw, list):
        queries = raw
    else:
        raise ValueError("Suite YAML must be a dict with 'queries' or a list")

    cases: list[QueryCase] = []
    for i, q in enumerate(queries):
        if isinstance(q, str):
            cases.append(QueryCase(id=f"q{i+1:02d}", query=q, tags=[], expect={}))
            continue
        if not isinstance(q, dict):
            raise ValueError(f"Invalid query entry at index {i}: expected string or object")
        qid = str(q.get("id") or f"q{i+1:02d}")
        text = q.get("query") or q.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Query '{qid}' is missing a non-empty 'query' field")
        tags = q.get("tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]
        expect = q.get("expect") or {}
        if not isinstance(expect, dict):
            expect = {}
        cases.append(QueryCase(id=qid, query=text.strip(), tags=[str(t) for t in tags], expect=expect))

    return suite_name, domain, cases


def _pctl(values: list[int], pct: float) -> int | None:
    if not values:
        return None
    xs = sorted(values)
    k = int(round((pct / 100.0) * (len(xs) - 1)))
    return xs[max(0, min(len(xs) - 1, k))]


def _pctl_float(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = int(round((pct / 100.0) * (len(xs) - 1)))
    return float(xs[max(0, min(len(xs) - 1, k))])


def _summarize_pipeline(rows: list[dict[str, Any]], pipeline: str) -> dict[str, Any]:
    ok: list[dict[str, Any]] = []
    errors = 0
    latencies: list[int] = []
    costs: list[float] = []
    retrieved_counts: list[int] = []

    halluc_flags: dict[str, int] = {}
    grounding_flags: dict[str, int] = {}

    for row in rows:
        result = (row.get("results") or {}).get(pipeline) or {}
        evaluation = (row.get("evaluations") or {}).get(pipeline) or {}

        if result.get("error") is not None:
            errors += 1
        else:
            ok.append(row)
            if isinstance(result.get("latency_ms"), int):
                latencies.append(int(result["latency_ms"]))
            if isinstance(result.get("cost_estimate_usd"), (int, float)):
                costs.append(float(result["cost_estimate_usd"]))

        if pipeline == "rag":
            chunks = result.get("retrieved_chunks") or []
            if isinstance(chunks, list):
                retrieved_counts.append(len(chunks))

        for f in evaluation.get("hallucination_flags") or []:
            halluc_flags[str(f)] = halluc_flags.get(str(f), 0) + 1
        for f in evaluation.get("grounding_flags") or []:
            grounding_flags[str(f)] = grounding_flags.get(str(f), 0) + 1

    summary: dict[str, Any] = {
        "pipeline": pipeline,
        "runs": len(rows),
        "ok": len(ok),
        "errors": errors,
        "latency_ms_avg": int(round(statistics.mean(latencies))) if latencies else None,
        "latency_ms_p50": _pctl(latencies, 50),
        "latency_ms_p95": _pctl(latencies, 95),
        "cost_usd_avg": float(statistics.mean(costs)) if costs else None,
        "cost_usd_total": float(sum(costs)) if costs else None,
        "hallucination_flags": halluc_flags,
        "grounding_flags": grounding_flags,
    }
    if pipeline == "rag":
        summary["retrieved_chunks_avg"] = float(statistics.mean(retrieved_counts)) if retrieved_counts else None
    return summary


def _rank_key(
    *,
    result: dict[str, Any],
    evaluation: dict[str, Any],
    expectation_score: float,
) -> tuple[float, int, int, float, int]:
    quality = evaluation.get("quality_score")
    base_quality = float(quality) if isinstance(quality, (int, float)) else 0.0
    combined = base_quality * float(expectation_score)

    halluc = evaluation.get("hallucination_flags") or []
    grounding = evaluation.get("grounding_flags") or []
    halluc_n = len(halluc) if isinstance(halluc, list) else 0
    grounding_n = len(grounding) if isinstance(grounding, list) else 0

    cost = result.get("cost_estimate_usd")
    cost_key = -float(cost) if isinstance(cost, (int, float)) else float("-inf")

    latency = result.get("latency_ms")
    latency_key = -int(latency) if isinstance(latency, int) else -10**12

    return combined, -grounding_n, -halluc_n, cost_key, latency_key


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MVP regression query suite against a running backend.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Backend base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--mode",
        default="docs",
        choices=["docs", "general"],
        help="Run mode: docs-grounded or general (default: docs)",
    )
    parser.add_argument(
        "--suite",
        default=str(Path("backend/tests/fixtures/mvp_queries.yaml")),
        help="Path to suite YAML (default: backend/tests/fixtures/mvp_queries.yaml)",
    )
    parser.add_argument("--domain", default=None, help="Override domain (default: suite value)")
    parser.add_argument(
        "--pipelines",
        default="prompt,rag",
        help="Comma-separated pipelines to run (default: prompt,rag)",
    )
    parser.add_argument(
        "--only-ids",
        default=None,
        help="Comma-separated case ids to run (default: run all cases in the suite)",
    )
    parser.add_argument(
        "--only-tags",
        default=None,
        help="Comma-separated tags to run (case must contain at least one) (default: run all cases)",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0, help="HTTP timeout per request")
    parser.add_argument("--max-queries", type=int, default=None, help="Run only the first N queries")
    parser.add_argument("--sleep-ms", type=int, default=0, help="Sleep between queries (rate limiting)")
    parser.add_argument("--out-jsonl", default=None, help="Optional path to write each RunResponse as JSONL")
    parser.add_argument("--judge", action="store_true", help="Use an LLM judge to score and rank answers")
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model (default: OPENAI_MODEL from backend settings)",
    )
    parser.add_argument(
        "--judge-answer-chars",
        type=int,
        default=2500,
        help="Max answer characters per pipeline to include in the judge prompt (default: 2500)",
    )
    parser.add_argument(
        "--judge-chunk-chars",
        type=int,
        default=800,
        help="Max chunk preview characters to include for RAG evidence (default: 800)",
    )
    parser.add_argument(
        "--show-expect",
        action="store_true",
        help="Print expectation mismatches per query/pipeline (only for cases with 'expect')",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs, do not call the backend")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit non-zero if any pipeline errors occur")
    args = parser.parse_args()

    base_url = str(args.base_url).rstrip("/")
    suite_path = Path(args.suite)
    suite_name, suite_domain, cases = _load_suite(suite_path)
    domain = args.domain or suite_domain or "fastapi_docs"
    pipelines = [p.strip() for p in str(args.pipelines).split(",") if p.strip()]
    if not pipelines:
        print("ERROR: --pipelines must be non-empty", file=sys.stderr)
        return 2

    if args.only_ids:
        wanted = {s.strip() for s in str(args.only_ids).split(",") if s.strip()}
        cases = [c for c in cases if c.id in wanted]
        if not cases:
            print(f"ERROR: no cases matched --only-ids ({sorted(wanted)})", file=sys.stderr)
            return 2

    if args.only_tags:
        wanted_tags = {s.strip() for s in str(args.only_tags).split(",") if s.strip()}
        cases = [c for c in cases if any(t in wanted_tags for t in (c.tags or []))]
        if not cases:
            print(f"ERROR: no cases matched --only-tags ({sorted(wanted_tags)})", file=sys.stderr)
            return 2

    if args.max_queries is not None:
        cases = cases[: max(0, int(args.max_queries))]

    if args.dry_run:
        print(f"[dry-run] suite={suite_name} domain={domain} pipelines={pipelines} queries={len(cases)}")
        for i, c in enumerate(cases, start=1):
            print(f"{i:02d} {c.id}: {c.query}")
        return 0

    status, health = _http_json(method="GET", url=f"{base_url}/health", timeout_s=args.timeout_s)
    if status != 200:
        print(f"ERROR: /health returned {status}: {health}", file=sys.stderr)
        return 2

    status, domains_resp = _http_json(method="GET", url=f"{base_url}/domains", timeout_s=args.timeout_s)
    if status != 200:
        print(f"ERROR: /domains returned {status}: {domains_resp}", file=sys.stderr)
        return 2
    available = set((domains_resp or {}).get("domains") or [])
    if domain not in available:
        print(f"ERROR: domain {domain!r} not in /domains: {sorted(available)}", file=sys.stderr)
        return 2

    out_f = None
    if args.out_jsonl:
        out_path = Path(args.out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("a", encoding="utf-8")

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    run_stamp = _utc_stamp()

    try:
        print(
            f"suite={suite_name} domain={domain} mode={args.mode} pipelines={pipelines} queries={len(cases)} base_url={base_url}"
        )

        winners_rank: dict[str, int] = {}
        expect_scores: dict[str, list[float]] = {p: [] for p in pipelines}
        expect_failures: dict[str, int] = {p: 0 for p in pipelines}
        judge_scores: dict[str, list[float]] = {p: [] for p in pipelines}
        judge_winners: dict[str, int] = {}

        for i, c in enumerate(cases, start=1):
            payload = {
                "domain": domain,
                "query": c.query,
                "mode": str(args.mode),
                "pipelines": pipelines,
                "expect": c.expect or None,
                "run_id": f"regression:{suite_name}:{run_stamp}:{i:03d}:{c.id}",
                "client_metadata": {"suite": suite_name, "case_id": c.id, "tags": c.tags},
            }
            if args.judge:
                payload["judge"] = True
                if args.judge_model:
                    payload["judge_model"] = str(args.judge_model)
                payload["judge_answer_chars"] = int(args.judge_answer_chars)
                payload["judge_chunk_chars"] = int(args.judge_chunk_chars)
            status, resp = _http_json(
                method="POST", url=f"{base_url}/run", payload=payload, timeout_s=args.timeout_s
            )
            if status != 200 or not isinstance(resp, dict):
                print(f"{i:02d}/{len(cases)} {c.id}: HTTP {status}", file=sys.stderr)
                continue

            rows.append(resp)
            if out_f is not None:
                out_f.write(json.dumps(resp) + "\n")

            rag_chunks = None
            rag = (resp.get("results") or {}).get("rag")
            if isinstance(rag, dict) and isinstance(rag.get("retrieved_chunks"), list):
                rag_chunks = len(rag["retrieved_chunks"])

            results = resp.get("results") or {}
            evals = resp.get("evaluations") or {}

            winner: str | None = None
            winner_ties: list[str] | None = None
            ranked: list[tuple[str, tuple[float, int, int, float, int]]] = []
            for p in pipelines:
                r = results.get(p) or {}
                e = evals.get(p) or {}
                ans = str(r.get("answer") or "")
                expect = dict(c.expect or {})
                if "trap" in c.tags and "expect_idk" not in expect:
                    expect["expect_idk"] = True
                exp_score, details = _check_expectations(ans, expect)
                if expect:
                    expect_scores[p].append(float(exp_score))
                    if exp_score < 1.0:
                        expect_failures[p] = expect_failures.get(p, 0) + 1
                        if args.show_expect:
                            snippet = " ".join(ans.strip().split())
                            snippet = snippet[:240] + ("…" if len(snippet) > 240 else "")
                            print(
                                f"  expect_mismatch case={c.id} pipeline={p} score={exp_score:.2f} details={details} answer_snippet={snippet!r}",
                                file=sys.stderr,
                            )
                ranked.append((p, _rank_key(result=r, evaluation=e, expectation_score=exp_score)))
            judge_label = None
            if args.judge:
                judged = resp.get("judge")
                if isinstance(judged, dict) and not judged.get("error"):
                    scores = judged.get("scores")
                    if isinstance(scores, dict):
                        for p in pipelines:
                            v = scores.get(p)
                            if isinstance(v, (int, float)):
                                judge_scores[p].append(float(v))
                    w = judged.get("winner")
                    if isinstance(w, str) and w:
                        judge_winners[w] = judge_winners.get(w, 0) + 1
                        judge_label = w

            if ranked:
                best_key = max(k for _, k in ranked)
                winners = sorted([p for p, k in ranked if k == best_key])
                if len(winners) == 1:
                    winner = winners[0]
                    winners_rank[winner] = winners_rank.get(winner, 0) + 1
                else:
                    winner_ties = winners
                    winners_rank["tie"] = winners_rank.get("tie", 0) + 1

            winner_label = winner if winner is not None else ("tie" if winner_ties else "?")
            judge_suffix = f" judge={judge_label}" if judge_label else ""
            print(f"{i:02d}/{len(cases)} {c.id}: ok (rag_chunks={rag_chunks}) winner={winner_label}{judge_suffix}")

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)
    finally:
        if out_f is not None:
            out_f.close()

    elapsed_s = time.perf_counter() - started
    if not rows:
        print("No successful responses collected.", file=sys.stderr)
        return 2

    winners: dict[str, int] = {}
    for row in rows:
        summary = row.get("summary_metrics") or {}
        w = summary.get("winner_by_quality")
        if w is not None:
            winners[str(w)] = winners.get(str(w), 0) + 1
        elif summary.get("winner_by_quality_ties"):
            winners["tie"] = winners.get("tie", 0) + 1

    summaries = [_summarize_pipeline(rows, p) for p in pipelines]
    for s in summaries:
        p = str(s.get("pipeline") or "")
        xs = expect_scores.get(p) or []
        if xs:
            s["expect_score_avg"] = float(statistics.mean(xs))
            s["expect_score_p50"] = _pctl_float(xs, 50)
            s["expect_failures"] = int(expect_failures.get(p, 0))
            s["expect_cases"] = int(len(xs))
        js = judge_scores.get(p) or []
        if js:
            s["judge_score_avg"] = float(statistics.mean(js))
            s["judge_score_p50"] = _pctl_float(js, 50)
            s["judge_cases"] = int(len(js))
    print("")
    print(f"completed={len(rows)}/{len(cases)} elapsed_s={elapsed_s:.1f}")
    print(f"winner_by_quality_counts_backend={winners}")
    print(f"winner_by_rank_counts={winners_rank}")
    if judge_winners:
        print(f"winner_by_judge_counts={judge_winners}")
    for s in summaries:
        print(json.dumps(s, ensure_ascii=False))

    if args.fail_on_error and any((s.get("errors") or 0) > 0 for s in summaries):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
