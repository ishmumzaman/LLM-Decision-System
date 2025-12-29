from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


RunMode = Literal["docs", "general"]


ExpectPattern = str
ExpectPatternGroup = list[ExpectPattern] | ExpectPattern


class ExpectationSpec(BaseModel):
    must_include: list[ExpectPattern] | ExpectPattern = Field(default_factory=list)
    must_not_include: list[ExpectPattern] | ExpectPattern = Field(default_factory=list)
    must_include_any: list[ExpectPatternGroup] | ExpectPatternGroup = Field(default_factory=list)
    expect_idk: bool = False


class RunRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    domain: str = Field(..., min_length=1)
    mode: RunMode = Field(
        "docs",
        description="Answering mode. 'docs' is docs-grounded; 'general' allows general knowledge.",
    )
    pipelines: list[str] = Field(default_factory=lambda: ["prompt", "rag"])
    expect: ExpectationSpec | None = None
    judge: bool = Field(False, description="If true, run an LLM judge to rank the pipeline outputs.")
    judge_model: str | None = Field(None, description="Optional judge model override.")
    judge_answer_chars: int = Field(2500, ge=200, le=20_000, description="Max answer chars per pipeline for judge.")
    judge_chunk_chars: int = Field(800, ge=100, le=5000, description="Max retrieved chunk chars for judge context.")
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


class JudgeResult(BaseModel):
    winner: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    criteria: dict[str, dict[str, float]] = Field(default_factory=dict)
    rationale: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_estimate_usd: float | None = None
    error: str | None = None


class RuleBreakdownItem(BaseModel):
    rule: str
    penalty: float = 0.0
    hallucination_flags: list[str] = Field(default_factory=list)
    grounding_flags: list[str] = Field(default_factory=list)
    note: str | None = None


class EvaluationResult(BaseModel):
    quality_score: float | None = None
    penalty_total: float | None = None
    rule_breakdown: list[RuleBreakdownItem] = Field(default_factory=list)
    expect_score: float | None = None
    expect_details: dict[str, Any] | None = None
    hallucination_flags: list[str] = Field(default_factory=list)
    grounding_flags: list[str] = Field(default_factory=list)
    notes: str | None = None


class RunResponse(BaseModel):
    run_id: str
    timestamp: datetime
    domain: str
    mode: RunMode
    query: str
    results: dict[str, PipelineResult]
    evaluations: dict[str, EvaluationResult]
    summary_metrics: dict[str, Any]
    judge: JudgeResult | None = None
