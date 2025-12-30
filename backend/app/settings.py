from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(repo_root() / "backend" / ".env", repo_root() / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider (MVP: OpenAI)
    llm_provider: str = Field("openai", validation_alias="LLM_PROVIDER")

    # OpenAI
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", validation_alias="OPENAI_MODEL")
    openai_judge_model: str = Field("gpt-4o", validation_alias="OPENAI_JUDGE_MODEL")
    openai_embedding_model: str = Field(
        "text-embedding-3-small", validation_alias="OPENAI_EMBEDDING_MODEL"
    )
    openai_finetuned_model: str | None = Field(default=None, validation_alias="OPENAI_FINETUNED_MODEL")

    # Generation defaults (fairness invariant: shared across pipelines per run)
    temperature: float = Field(0.2, validation_alias="GEN_TEMPERATURE")
    top_p: float = Field(1.0, validation_alias="GEN_TOP_P")
    max_output_tokens: int = Field(800, validation_alias="GEN_MAX_OUTPUT_TOKENS")

    # RAG (global caps)
    rag_max_context_chars: int = Field(12_000, validation_alias="RAG_MAX_CONTEXT_CHARS")
    rag_max_chunks: int = Field(8, validation_alias="RAG_MAX_CHUNKS")

    # Timeouts
    pipeline_timeout_s: float = Field(30.0, validation_alias="PIPELINE_TIMEOUT_S")

    # Paths
    domains_dir: Path = Field(default_factory=lambda: repo_root() / "domains")
    domains_registry_path: Path = Field(default_factory=lambda: repo_root() / "domains" / "registry.yaml")
    run_log_path: Path = Field(default_factory=lambda: repo_root() / "backend" / "runs.jsonl")
    regression_suites_dir: Path = Field(
        default_factory=lambda: repo_root() / "backend" / "tests" / "fixtures",
        validation_alias="REGRESSION_SUITES_DIR",
    )

    # Demo mode (optional): allow running the UI/backend without an API key by replaying sample runs.
    demo_mode: bool = Field(False, validation_alias="DEMO_MODE")
    demo_runs_path: Path = Field(
        default_factory=lambda: repo_root() / "backend" / "demo_runs.json",
        validation_alias="DEMO_RUNS_PATH",
    )

    # Optional safety budget (best-effort; enforced only when cost estimation is enabled)
    max_run_cost_usd: float | None = Field(default=None, validation_alias="MAX_RUN_COST_USD")

    # Optional cost estimation (USD per 1M tokens)
    openai_cost_input_per_1m: float | None = Field(
        default=None, validation_alias="OPENAI_COST_INPUT_PER_1M"
    )
    openai_cost_output_per_1m: float | None = Field(
        default=None, validation_alias="OPENAI_COST_OUTPUT_PER_1M"
    )
    openai_cost_input_per_1m_finetuned: float | None = Field(
        default=None, validation_alias="OPENAI_COST_INPUT_PER_1M_FINETUNED"
    )
    openai_cost_output_per_1m_finetuned: float | None = Field(
        default=None, validation_alias="OPENAI_COST_OUTPUT_PER_1M_FINETUNED"
    )

    @field_validator(
        "openai_finetuned_model",
        "openai_cost_input_per_1m",
        "openai_cost_output_per_1m",
        "openai_cost_input_per_1m_finetuned",
        "openai_cost_output_per_1m_finetuned",
        "max_run_cost_usd",
        mode="before",
    )
    @classmethod
    def _empty_str_to_none(cls, v: object) -> object:
        if v == "":
            return None
        return v
