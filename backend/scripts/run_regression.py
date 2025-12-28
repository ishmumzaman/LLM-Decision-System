from __future__ import annotations

import argparse
import json
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
            cases.append(QueryCase(id=f"q{i+1:02d}", query=q, tags=[]))
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
        cases.append(QueryCase(id=qid, query=text.strip(), tags=[str(t) for t in tags]))

    return suite_name, domain, cases


def _pctl(values: list[int], pct: float) -> int | None:
    if not values:
        return None
    xs = sorted(values)
    k = int(round((pct / 100.0) * (len(xs) - 1)))
    return xs[max(0, min(len(xs) - 1, k))]


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
        "hallucination_flags": halluc_flags,
        "grounding_flags": grounding_flags,
    }
    if pipeline == "rag":
        summary["retrieved_chunks_avg"] = float(statistics.mean(retrieved_counts)) if retrieved_counts else None
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MVP regression query suite against a running backend.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Backend base URL (default: http://127.0.0.1:8000)",
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
    parser.add_argument("--timeout-s", type=float, default=120.0, help="HTTP timeout per request")
    parser.add_argument("--max-queries", type=int, default=None, help="Run only the first N queries")
    parser.add_argument("--sleep-ms", type=int, default=0, help="Sleep between queries (rate limiting)")
    parser.add_argument("--out-jsonl", default=None, help="Optional path to write each RunResponse as JSONL")
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
        print(f"suite={suite_name} domain={domain} pipelines={pipelines} queries={len(cases)} base_url={base_url}")

        for i, c in enumerate(cases, start=1):
            payload = {
                "domain": domain,
                "query": c.query,
                "pipelines": pipelines,
                "run_id": f"regression:{suite_name}:{run_stamp}:{i:03d}:{c.id}",
                "client_metadata": {"suite": suite_name, "case_id": c.id, "tags": c.tags},
            }
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

            print(f"{i:02d}/{len(cases)} {c.id}: ok (rag_chunks={rag_chunks})")

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
    print("")
    print(f"completed={len(rows)}/{len(cases)} elapsed_s={elapsed_s:.1f}")
    print(f"winner_by_quality_counts={winners}")
    for s in summaries:
        print(json.dumps(s, ensure_ascii=False))

    if args.fail_on_error and any((s.get("errors") or 0) > 0 for s in summaries):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
