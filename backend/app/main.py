from __future__ import annotations

import asyncio
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
import yaml

from app.domains.loader import load_domain_spec, load_registry
from app.eval.expectations import check_expectations
from app.eval.evaluator import evaluate_results
from app.eval.judge import judge_run
from app.llm.openai_client import build_openai_client
from app.pipelines.finetuned import run_finetuned
from app.pipelines.prompt_only import run_prompt_only
from app.pipelines.rag import run_rag
from app.schemas.run import PipelineResult, RunRequest, RunResponse
from app.settings import Settings
from app.logging.run_logger import append_run


settings = Settings()
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client  # noqa: PLW0603
    if _client is None:
        _client = build_openai_client(settings)
    return _client

app = FastAPI(title="LLM Decision System", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/domains")
async def list_domains() -> dict[str, list[str]]:
    return {"domains": load_registry(settings.domains_registry_path)}


_SUITE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _load_suite_file(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))  # type: ignore[no-untyped-call]
    if raw is None:
        return {"suite": path.stem, "queries": []}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return {"suite": path.stem, "queries": raw}
    raise ValueError("Suite YAML must be a dict with 'queries' or a list")


@app.get("/suites")
async def list_suites() -> dict[str, list[dict[str, Any]]]:
    suites: list[dict[str, Any]] = []
    suites_dir = settings.regression_suites_dir
    if not suites_dir.exists():
        return {"suites": suites}

    for path in sorted([p for p in suites_dir.glob("*.yaml") if p.is_file()]):
        try:
            data = _load_suite_file(path)
        except Exception:  # noqa: BLE001
            continue
        queries = data.get("queries") or []
        suites.append(
            {
                "id": path.stem,
                "suite": str(data.get("suite") or path.stem),
                "domain": data.get("domain"),
                "description": data.get("description"),
                "cases": len(queries) if isinstance(queries, list) else 0,
            }
        )

    return {"suites": suites}


@app.get("/suites/{suite_id}")
async def get_suite(suite_id: str) -> dict[str, Any]:
    if _SUITE_ID_RE.match(suite_id) is None:
        raise HTTPException(status_code=400, detail={"error": "INVALID_SUITE_ID"})

    path = settings.regression_suites_dir / f"{suite_id}.yaml"
    if not path.exists():
        raise HTTPException(status_code=404, detail={"error": "SUITE_NOT_FOUND", "suite": suite_id})

    data = _load_suite_file(path)
    queries = data.get("queries") or []
    cases: list[dict[str, Any]] = []
    if isinstance(queries, list):
        for i, q in enumerate(queries):
            if isinstance(q, str):
                cases.append({"id": f"q{i+1:02d}", "query": q, "tags": [], "expect": {}})
                continue
            if not isinstance(q, dict):
                continue
            qid = str(q.get("id") or f"q{i+1:02d}")
            text = q.get("query") or q.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            tags = q.get("tags") or []
            if not isinstance(tags, list):
                tags = [tags]
            expect = q.get("expect") or {}
            if not isinstance(expect, dict):
                expect = {}
            cases.append(
                {"id": qid, "query": text.strip(), "tags": [str(t) for t in tags], "expect": expect}
            )

    return {
        "id": suite_id,
        "suite": str(data.get("suite") or suite_id),
        "domain": data.get("domain"),
        "description": data.get("description"),
        "queries": cases,
    }


async def _run_one(pipeline: str, request: RunRequest, domain_name: str) -> PipelineResult:
    domain = load_domain_spec(settings.domains_dir, domain_name)
    if pipeline == "prompt":
        return await run_prompt_only(
            client=_get_client(),
            settings=settings,
            domain=domain,
            query=request.query,
            mode=request.mode,
        )
    if pipeline == "rag":
        return await run_rag(
            client=_get_client(),
            settings=settings,
            domain=domain,
            query=request.query,
            mode=request.mode,
        )
    if pipeline == "finetune":
        return await run_finetuned(
            client=_get_client(),
            settings=settings,
            domain=domain,
            query=request.query,
            mode=request.mode,
        )
    raise ValueError(f"Unsupported pipeline: {pipeline}")


def _expected_model_for_pipeline(pipeline: str, domain_name: str) -> str:
    if pipeline != "finetune":
        return settings.openai_model
    domain = load_domain_spec(settings.domains_dir, domain_name)
    return domain.finetuned_model or settings.openai_finetuned_model or settings.openai_model


@app.post("/run", response_model=RunResponse)
async def run(request: RunRequest) -> RunResponse:
    available = set(load_registry(settings.domains_registry_path))
    if request.domain not in available:
        raise HTTPException(status_code=400, detail={"error": "INVALID_DOMAIN", "domains": sorted(available)})
    if settings.llm_provider != "openai":
        raise HTTPException(
            status_code=500,
            detail={"error": "UNSUPPORTED_LLM_PROVIDER", "provider": settings.llm_provider},
        )
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=500,
            detail={"error": "MISSING_OPENAI_API_KEY", "hint": "Set OPENAI_API_KEY (or backend/.env)."},
        )

    run_id = request.run_id or str(uuid.uuid4())
    started = time.perf_counter()

    pipelines = request.pipelines or ["prompt", "rag"]
    supported = {"prompt", "rag", "finetune"}
    for p in pipelines:
        if p not in supported:
            raise HTTPException(status_code=400, detail={"error": "INVALID_PIPELINE", "pipeline": p})

    tasks: dict[str, asyncio.Task[PipelineResult]] = {
        p: asyncio.create_task(_run_one(p, request, request.domain)) for p in pipelines
    }

    results: dict[str, PipelineResult] = {}
    for name, task in tasks.items():
        try:
            results[name] = await asyncio.wait_for(task, timeout=settings.pipeline_timeout_s)
        except asyncio.TimeoutError:
            latency_ms = int((time.perf_counter() - started) * 1000)
            results[name] = PipelineResult(
                pipeline=name,
                model=_expected_model_for_pipeline(name, request.domain),
                generation_config={
                    "temperature": settings.temperature,
                    "top_p": settings.top_p,
                    "max_tokens": settings.max_output_tokens,
                },
                answer="",
                latency_ms=latency_ms,
                tokens_in=None,
                tokens_out=None,
                cost_estimate_usd=None,
                retrieved_chunks=None,
                flags={},
                error="TIMEOUT",
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - started) * 1000)
            results[name] = PipelineResult(
                pipeline=name,
                model=_expected_model_for_pipeline(name, request.domain),
                generation_config={
                    "temperature": settings.temperature,
                    "top_p": settings.top_p,
                    "max_tokens": settings.max_output_tokens,
                },
                answer="",
                latency_ms=latency_ms,
                tokens_in=None,
                tokens_out=None,
                cost_estimate_usd=None,
                retrieved_chunks=None,
                flags={},
                error=f"ERROR: {type(exc).__name__}",
            )

    if settings.max_run_cost_usd is not None:
        for result in results.values():
            if result.cost_estimate_usd is None:
                continue
            if result.cost_estimate_usd > settings.max_run_cost_usd:
                result.flags["COST_CAP_EXCEEDED"] = {
                    "cap_usd": settings.max_run_cost_usd,
                    "estimated_usd": result.cost_estimate_usd,
                }

    domain_spec = load_domain_spec(settings.domains_dir, request.domain)
    rule_names = list(domain_spec.evaluation_rules)
    if request.mode == "general":
        rule_names = [r for r in rule_names if r != "not_grounded_check"]
    evaluations, summary = evaluate_results(results=results, rule_names=rule_names)

    tags: list[str] = []
    if request.client_metadata and isinstance(request.client_metadata.get("tags"), list):
        tags = [str(t) for t in request.client_metadata.get("tags") if t is not None]

    expect_dict = request.expect.model_dump(exclude_none=True) if request.expect is not None else None
    expect_active = False
    if expect_dict is not None:
        expect_active = bool(
            expect_dict.get("must_include")
            or expect_dict.get("must_not_include")
            or expect_dict.get("must_include_any")
            or expect_dict.get("expect_idk")
            or ("trap" in tags)
        )

    if expect_active and expect_dict is not None:
        for p, result in results.items():
            if result.error is not None:
                continue
            score, details = check_expectations(answer=result.answer, expect=expect_dict, tags=tags)
            evaluations[p] = evaluations[p].model_copy(
                update={
                    "expect_score": float(score),
                    "expect_details": details or None,
                }
            )

        scored = {k: v.expect_score for k, v in evaluations.items() if v.expect_score is not None}
        if scored:
            best = max(float(s) for s in scored.values() if s is not None)
            winners = sorted([k for k, s in scored.items() if float(s) == best])
            if len(winners) == 1:
                summary["winner_by_expect"] = winners[0]
            else:
                summary["winner_by_expect"] = None
                summary["winner_by_expect_ties"] = winners

    judge = None
    if request.judge:
        judge = await judge_run(
            client=_get_client(),
            settings=settings,
            mode=request.mode,
            domain=request.domain,
            query=request.query,
            results=results,
            judge_model=request.judge_model,
            judge_answer_chars=int(request.judge_answer_chars),
            judge_chunk_chars=int(request.judge_chunk_chars),
        )
        if judge.error is None:
            scored = {k: float(v) for k, v in (judge.scores or {}).items() if isinstance(v, (int, float))}
            if judge.winner and judge.winner != "tie":
                summary["winner_by_judge"] = judge.winner
            elif scored:
                ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
                top_score = ranked[0][1]
                near_top = [p for p, s in ranked if (top_score - s) <= 0.5]
                if len(near_top) < 2 and len(ranked) >= 2:
                    near_top = [ranked[0][0], ranked[1][0]]
                summary["winner_by_judge"] = None
                summary["winner_by_judge_ties"] = sorted(near_top)

    response = RunResponse(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc),
        domain=request.domain,
        mode=request.mode,
        query=request.query,
        results=results,
        evaluations=evaluations,
        summary_metrics=summary,
        judge=judge,
    )
    append_run(settings.run_log_path, response)
    return response
