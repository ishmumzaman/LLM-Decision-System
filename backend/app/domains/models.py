from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field


class DomainConfig(BaseModel):
    name: str
    documents_path: str
    chunk_size: int = Field(..., gt=0)
    chunk_overlap: int = Field(0, ge=0)
    retrieval_k: int = Field(5, gt=0)
    evaluation_rules: list[str] = Field(default_factory=list)
    domain_prompt_prefix: str | None = None
    finetuned_model: str | None = None
    allowed_file_types: list[str] = Field(default_factory=lambda: ["md", "txt"])
    max_docs: int | None = Field(default=None, gt=0)


@dataclass(frozen=True)
class DomainSpec:
    name: str
    documents_dir: Path
    artifacts_dir: Path

    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    evaluation_rules: list[str]
    domain_prompt_prefix: str | None
    finetuned_model: str | None
    allowed_file_types: list[str]
    max_docs: int | None

    index_path: Path
    chunks_path: Path
    index_meta_path: Path
