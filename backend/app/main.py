from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

from app.domains.loader import load_domain_spec, load_registry
from app.eval.evaluator import evaluate_results
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


async def _run_one(pipeline: str, request: RunRequest, domain_name: str) -> PipelineResult:
    domain = load_domain_spec(settings.domains_dir, domain_name)
    if pipeline == "prompt":
        return await run_prompt_only(
            client=_get_client(), settings=settings, domain=domain, query=request.query
        )
    if pipeline == "rag":
        return await run_rag(client=_get_client(), settings=settings, domain=domain, query=request.query)
    if pipeline == "finetune":
        return await run_finetuned(
            client=_get_client(), settings=settings, domain=domain, query=request.query
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
    evaluations, summary = evaluate_results(results=results, rule_names=domain_spec.evaluation_rules)

    response = RunResponse(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc),
        domain=request.domain,
        query=request.query,
        results=results,
        evaluations=evaluations,
        summary_metrics=summary,
    )
    append_run(settings.run_log_path, response)
    return response
