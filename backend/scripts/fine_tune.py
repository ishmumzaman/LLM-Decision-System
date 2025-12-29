from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI


BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.domains.loader import load_domain_spec  # noqa: E402
from app.pipelines.common import system_prompt_parts  # noqa: E402
from app.settings import Settings  # noqa: E402


_GPT4O_MINI_SNAPSHOT_RE = re.compile(r"^gpt-4o-mini-\\d{4}-\\d{2}-\\d{2}$")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_dataset_yaml(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))  # type: ignore[no-untyped-call]
    if isinstance(raw, list):
        return {"dataset": path.stem}, [dict(x) for x in raw]
    if isinstance(raw, dict):
        examples = raw.get("examples") or raw.get("data") or raw.get("items") or []
        if not isinstance(examples, list):
            raise ValueError("Dataset YAML must contain an 'examples: [...]' list")
        meta = {k: v for k, v in raw.items() if k not in {"examples", "data", "items"}}
        return meta, [dict(x) for x in examples]
    raise ValueError("Dataset YAML must be a list or a dict with 'examples'")


def _build_system_prompt(domain_prompt_prefix: str | None) -> str:
    return "\n".join(system_prompt_parts(domain_prompt_prefix))


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


def _read_chunks_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("chunks.json must be a JSON array")
    out: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            out.append(item)
    return out


def cmd_prepare(args: argparse.Namespace) -> int:
    settings = Settings()
    domain = load_domain_spec(settings.domains_dir, str(args.domain))

    in_path = Path(args.in_yaml)
    meta, examples = _load_dataset_yaml(in_path)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = _build_system_prompt(domain.domain_prompt_prefix)

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(examples):
            if "messages" in ex:
                row = {"messages": ex["messages"]}
            else:
                query = ex.get("query") or ex.get("question")
                answer = ex.get("answer") or ex.get("completion") or ex.get("output")
                if not isinstance(query, str) or not query.strip():
                    raise ValueError(f"Example {i} missing non-empty 'query'")
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError(f"Example {i} missing non-empty 'answer'")
                row = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query.strip()},
                        {"role": "assistant", "content": answer.strip()},
                    ]
                }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    dataset_name = str(meta.get("dataset") or in_path.stem)
    print(f"prepared dataset={dataset_name} domain={domain.name} examples={written} -> {out_path.as_posix()}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    settings = Settings()
    if not settings.openai_api_key:
        print("ERROR: missing OPENAI_API_KEY", file=sys.stderr)
        return 2

    domain = load_domain_spec(settings.domains_dir, str(args.domain))
    chunks = _read_chunks_json(domain.chunks_path)

    skip_substrings = [s.strip() for s in (args.skip_source_substring or []) if s.strip()]
    if skip_substrings:
        chunks = [c for c in chunks if str(c.get("source") or "") and all(s not in str(c.get("source") or "") for s in skip_substrings)]

    if not chunks:
        print("ERROR: no chunks available after filtering", file=sys.stderr)
        return 2

    seed = int(args.seed) if args.seed is not None else random.randrange(0, 1_000_000_000)
    rng = random.Random(seed)

    count = int(args.count)
    picks = [rng.randrange(0, len(chunks)) for _ in range(count)]

    client = OpenAI(api_key=settings.openai_api_key)
    gen_model = str(args.gen_model or settings.openai_model)

    examples: list[dict[str, Any]] = []
    sys_prompt = "\n".join(
        [
            "You generate training examples for fine-tuning a model to answer FastAPI documentation questions.",
            "Given a documentation excerpt, create ONE realistic user question that can be answered using ONLY the excerpt.",
            "Then write the answer. Do not mention the excerpt or citations.",
            "Output MUST be a single JSON object: {\"query\": \"...\", \"answer\": \"...\"}. No extra text.",
        ]
    )

    for i, idx in enumerate(picks, start=1):
        c = chunks[idx]
        excerpt = str(c.get("text") or "")
        if not excerpt.strip():
            continue

        user = "\n".join(
            [
                f"Excerpt (source={c.get('source')}, chunk_id={c.get('chunk_id')}):",
                excerpt.strip(),
            ]
        )

        resp = client.chat.completions.create(
            model=gen_model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            top_p=1.0,
            max_tokens=400,
        )
        content = resp.choices[0].message.content or ""
        try:
            obj = _extract_json_object(content)
        except Exception:  # noqa: BLE001
            continue

        query = obj.get("query")
        answer = obj.get("answer")
        if not isinstance(query, str) or not query.strip():
            continue
        if not isinstance(answer, str) or not answer.strip():
            continue

        examples.append(
            {
                "id": f"gen_{i:04d}",
                "query": query.strip(),
                "answer": answer.strip(),
                "source": c.get("source"),
                "chunk_id": c.get("chunk_id"),
            }
        )

        if args.sleep_ms and int(args.sleep_ms) > 0:
            time.sleep(int(args.sleep_ms) / 1000.0)

    out_path = Path(args.out_yaml)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_name = str(args.dataset or out_path.stem)
    doc = {
        "dataset": dataset_name,
        "domain": domain.name,
        "generated_at": _utc_stamp(),
        "generator_model": gen_model,
        "seed": seed,
        "examples": examples,
    }
    out_path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")  # type: ignore[no-untyped-call]

    print(f"generated dataset={dataset_name} domain={domain.name} examples={len(examples)} -> {out_path.as_posix()}")
    return 0


def _upload_jsonl(client: OpenAI, path: Path) -> str:
    with path.open("rb") as f:
        obj = client.files.create(file=f, purpose="fine-tune")
    return str(obj.id)


def _default_finetune_base_model(client: OpenAI, requested: str) -> str:
    if _GPT4O_MINI_SNAPSHOT_RE.fullmatch(requested):
        return requested
    if not requested.startswith("gpt-4o-mini"):
        return requested

    try:
        models = client.models.list()
    except Exception:  # noqa: BLE001
        return "gpt-4o-mini-2024-07-18"

    candidates: list[str] = []
    for m in getattr(models, "data", []) or []:
        mid = getattr(m, "id", None)
        if isinstance(mid, str) and _GPT4O_MINI_SNAPSHOT_RE.fullmatch(mid):
            candidates.append(mid)

    return max(candidates) if candidates else "gpt-4o-mini-2024-07-18"


def cmd_start(args: argparse.Namespace) -> int:
    settings = Settings()
    if not settings.openai_api_key:
        print("ERROR: missing OPENAI_API_KEY", file=sys.stderr)
        return 2

    train_path = Path(args.training_jsonl)
    if not train_path.exists():
        print(f"ERROR: training file not found: {train_path}", file=sys.stderr)
        return 2

    client = OpenAI(api_key=settings.openai_api_key)
    train_file_id = _upload_jsonl(client, train_path)

    val_file_id = None
    if args.validation_jsonl:
        val_path = Path(args.validation_jsonl)
        if not val_path.exists():
            print(f"ERROR: validation file not found: {val_path}", file=sys.stderr)
            return 2
        val_file_id = _upload_jsonl(client, val_path)

    requested_model = str(args.base_model or settings.openai_model)
    base_model = requested_model if args.base_model else _default_finetune_base_model(client, requested_model)
    job_kwargs: dict[str, Any] = {"model": base_model, "training_file": train_file_id}
    if val_file_id is not None:
        job_kwargs["validation_file"] = val_file_id
    if args.suffix:
        job_kwargs["suffix"] = str(args.suffix)
    if args.n_epochs:
        n_epochs: Any
        if str(args.n_epochs).strip().lower() == "auto":
            n_epochs = "auto"
        else:
            n_epochs = int(args.n_epochs)
        job_kwargs["hyperparameters"] = {"n_epochs": n_epochs}

    job = client.fine_tuning.jobs.create(**job_kwargs)
    print(json.dumps(job.model_dump(), indent=2, ensure_ascii=False))

    if args.wait:
        job_id = str(job.id)
        while True:
            fresh = client.fine_tuning.jobs.retrieve(job_id)
            status = str(fresh.status)
            print(f"status={status} job_id={job_id}")
            if status in {"succeeded", "failed", "cancelled"}:
                print(json.dumps(fresh.model_dump(), indent=2, ensure_ascii=False))
                break
            time.sleep(float(args.poll_s))

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    settings = Settings()
    if not settings.openai_api_key:
        print("ERROR: missing OPENAI_API_KEY", file=sys.stderr)
        return 2

    client = OpenAI(api_key=settings.openai_api_key)
    job = client.fine_tuning.jobs.retrieve(str(args.job_id))
    print(json.dumps(job.model_dump(), indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tuning utilities (dataset -> JSONL -> OpenAI job).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare", help="Convert dataset YAML to OpenAI JSONL format.")
    p_prepare.add_argument("--domain", required=True, help="Domain name (e.g. fastapi_docs)")
    p_prepare.add_argument("--in-yaml", required=True, help="Input dataset YAML")
    p_prepare.add_argument("--out-jsonl", required=True, help="Output JSONL path")
    p_prepare.set_defaults(func=cmd_prepare)

    p_generate = sub.add_parser("generate", help="Generate a dataset YAML from domain chunks using an LLM.")
    p_generate.add_argument("--domain", required=True, help="Domain name (e.g. fastapi_docs)")
    p_generate.add_argument("--count", type=int, default=50, help="Number of examples to attempt (default: 50)")
    p_generate.add_argument("--out-yaml", required=True, help="Output dataset YAML path")
    p_generate.add_argument("--dataset", default=None, help="Dataset name override (default: file stem)")
    p_generate.add_argument("--gen-model", default=None, help="Generator model (default: OPENAI_MODEL)")
    p_generate.add_argument("--seed", type=int, default=None, help="Random seed for chunk sampling")
    p_generate.add_argument(
        "--skip-source-substring",
        action="append",
        default=["_llm-test.md"],
        help="Skip chunks whose source contains this substring (repeatable)",
    )
    p_generate.add_argument("--sleep-ms", type=int, default=0, help="Sleep between calls (rate limiting)")
    p_generate.set_defaults(func=cmd_generate)

    p_start = sub.add_parser("start", help="Upload JSONL and start an OpenAI fine-tuning job.")
    p_start.add_argument("--training-jsonl", required=True, help="Training JSONL path")
    p_start.add_argument("--validation-jsonl", default=None, help="Optional validation JSONL path")
    p_start.add_argument("--base-model", default=None, help="Base model to fine-tune (default: OPENAI_MODEL)")
    p_start.add_argument("--suffix", default=None, help="Optional job suffix label")
    p_start.add_argument("--n-epochs", default=None, help="Hyperparameter n_epochs (int or 'auto')")
    p_start.add_argument("--wait", action="store_true", help="Poll job until completion")
    p_start.add_argument("--poll-s", type=float, default=10.0, help="Polling interval seconds (default: 10)")
    p_start.set_defaults(func=cmd_start)

    p_status = sub.add_parser("status", help="Fetch an OpenAI fine-tuning job status by id.")
    p_status.add_argument("--job-id", required=True, help="Fine-tuning job id")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
