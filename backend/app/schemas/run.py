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


class CaseMetadata(BaseModel):
    id: str | None = None
    tags: list[str] = Field(default_factory=list)
    answerable_from_general_knowledge: bool | None = None
    requires_docs: bool | None = None
    expected_abstain_in_docs: bool | None = None


class RunRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    domain: str = Field(..., min_length=1)
    mode: RunMode = Field(
        "docs",
        description="Answering mode. 'docs' is docs-grounded; 'general' allows general knowledge.",
    )
    pipelines: list[str] = Field(default_factory=lambda: ["prompt", "rag"])
    expect: ExpectationSpec | None = None
    case: CaseMetadata | None = None
    judge: bool = Field(False, description="If true, run an LLM judge to rank the pipeline outputs.")
    judge_model: str | None = Field(None, description="Optional judge model override.")
    proxy_evidence: bool = Field(
        False, description="If true, run an LLM-based evidence support check (docs mode only)."
    )
    proxy_answerability: bool = Field(
        False, description="If true, run an LLM-based 'answerable without docs' estimation."
    )
    judge_answer_chars: int = Field(2500, ge=200, le=20_000, description="Max answer chars per pipeline for judge.")
    judge_chunk_chars: int = Field(800, ge=100, le=5000, description="Max retrieved chunk chars for judge context.")
    scoring: dict[str, Any] | None = Field(
        None, description="Optional client-provided scoring configuration (for transparency/audit)."
    )
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


ProxyAnswerabilityLabel = Literal["general", "requires_docs", "unsupported", "unknown"]


class AnswerabilityProxy(BaseModel):
    label: ProxyAnswerabilityLabel | None = None
    answerable_without_docs: bool | None = None
    confidence: float | None = None
    rationale: str | None = None
    error: str | None = None


class EvidenceSupportProxy(BaseModel):
    support_score: float | None = None
    unsupported_claims: list[str] = Field(default_factory=list)
    rationale: str | None = None
    error: str | None = None


class ProxiesResult(BaseModel):
    answerability: AnswerabilityProxy | None = None
    evidence_support: dict[str, EvidenceSupportProxy] = Field(default_factory=dict)
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
    abstained: bool | None = None
    abstention_expected: bool | None = None
    abstention_correct: bool | None = None
    abstention_score: float | None = None
    hallucination_flags: list[str] = Field(default_factory=list)
    grounding_flags: list[str] = Field(default_factory=list)
    notes: str | None = None


class RunResponse(BaseModel):
    run_id: str
    timestamp: datetime
    domain: str
    mode: RunMode
    query: str
    case: CaseMetadata | None = None
    scoring: dict[str, Any] | None = None
    results: dict[str, PipelineResult]
    evaluations: dict[str, EvaluationResult]
    summary_metrics: dict[str, Any]
    judge: JudgeResult | None = None
    proxies: ProxiesResult | None = None
