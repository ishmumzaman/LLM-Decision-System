from __future__ import annotations

from pathlib import Path

import yaml

from app.domains.models import DomainConfig, DomainSpec


def load_registry(registry_path: Path) -> list[str]:
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    domains = data.get("domains", [])
    if not isinstance(domains, list):
        raise ValueError("domains/registry.yaml must contain a top-level 'domains: [...]' list")
    return [str(d) for d in domains]


def load_domain_spec(domains_dir: Path, domain_name: str) -> DomainSpec:
    domain_dir = (domains_dir / domain_name).resolve()
    config_path = domain_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing domain config: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = DomainConfig.model_validate(raw)

    repo_root = domains_dir.resolve().parent
    documents_dir = (repo_root / config.documents_path).resolve()
    artifacts_dir = (domains_dir / domain_name / "artifacts").resolve()

    index_path = artifacts_dir / "index.faiss"
    chunks_path = artifacts_dir / "chunks.json"
    index_meta_path = artifacts_dir / "index_meta.json"

    return DomainSpec(
        name=config.name,
        documents_dir=documents_dir,
        artifacts_dir=artifacts_dir,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        retrieval_k=config.retrieval_k,
        evaluation_rules=list(config.evaluation_rules),
        domain_prompt_prefix=config.domain_prompt_prefix,
        finetuned_model=config.finetuned_model,
        allowed_file_types=[ft.lstrip(".") for ft in config.allowed_file_types],
        max_docs=config.max_docs,
        index_path=index_path,
        chunks_path=chunks_path,
        index_meta_path=index_meta_path,
    )
