from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    domain: str = Field(..., min_length=1)
    pipelines: list[str] = Field(default_factory=lambda: ["prompt", "rag"])
    run_id: str | None = None
    client_metadata: dict[str, Any] | None = None


class RetrievedChunk(BaseModel):
    chunk_id: int
    source: str
    text_preview: str
    score: float | None = None


class PipelineResult(BaseModel):
    pipeline: str
    model: str
    generation_config: dict[str, Any]
    answer: str
    latency_ms: int
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_estimate_usd: float | None = None
    retrieved_chunks: list[RetrievedChunk] | None = None
    flags: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class EvaluationResult(BaseModel):
    quality_score: float | None = None
    hallucination_flags: list[str] = Field(default_factory=list)
    grounding_flags: list[str] = Field(default_factory=list)
    notes: str | None = None


class RunResponse(BaseModel):
    run_id: str
    timestamp: datetime
    domain: str
    query: str
    results: dict[str, PipelineResult]
    evaluations: dict[str, EvaluationResult]
    summary_metrics: dict[str, Any]

