from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.schemas.run import PipelineResult


@dataclass(frozen=True)
class DemoSample:
    domain: str
    query: str
    results: dict[str, PipelineResult]


def _load_demo_file(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        samples = raw.get("samples") or []
        if isinstance(samples, list):
            return samples
    raise ValueError("demo_runs.json must be a list or a dict with 'samples: [...]'")


def load_demo_samples(path: Path) -> list[DemoSample]:
    if not path.exists():
        raise FileNotFoundError(f"Demo runs file not found: {path}")
    items = _load_demo_file(path)
    samples: list[DemoSample] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        domain = item.get("domain")
        query = item.get("query")
        results = item.get("results")
        if not isinstance(domain, str) or not domain:
            continue
        if not isinstance(query, str) or not query:
            continue
        if not isinstance(results, dict) or not results:
            continue
        parsed: dict[str, PipelineResult] = {}
        for k, v in results.items():
            if not isinstance(k, str) or not isinstance(v, dict):
                continue
            parsed[k] = PipelineResult.model_validate(v)
        if not parsed:
            continue
        samples.append(DemoSample(domain=domain, query=query, results=parsed))
    if not samples:
        raise ValueError(f"No usable demo samples in {path}")
    return samples


def pick_demo_sample(*, samples: list[DemoSample], domain: str) -> DemoSample:
    for s in samples:
        if s.domain == domain:
            return s
    return samples[0]

