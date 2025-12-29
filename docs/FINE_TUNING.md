# Fine-tuning (Optional)

The MVP compares `prompt` vs `rag`. This phase adds a third pipeline: `finetune`, which calls a fine-tuned OpenAI model with the same prompt template and generation settings used by `prompt`.

## What to optimize for
- Use a **separate eval set** (don’t train on your regression/eval queries).
- Keep the system prompt stable across `prompt` and `finetune` so the comparison is meaningful.
- Start small (100–500 examples), measure deltas vs cost/effort, then iterate.

## Workflow (repo-supported)
### 1) Create a dataset
Options:
- Curate a dataset YAML manually (recommended for quality).
- Auto-generate a dataset YAML from the domain chunks (fast, noisier):
  - `.\.venv\Scripts\python backend/scripts/fine_tune.py generate --domain fastapi_docs --count 200 --out-yaml backend/finetune/datasets/fastapi_docs_train_gen.yaml`

### 2) Convert dataset YAML -> JSONL
- `.\.venv\Scripts\python backend/scripts/fine_tune.py prepare --domain fastapi_docs --in-yaml backend/finetune/datasets/fastapi_docs_train_v1.yaml --out-jsonl backend/finetune/out/fastapi_docs_train_v1.jsonl`

The generated JSONL is intentionally untracked (`*.jsonl` is in `.gitignore`).

### 3) Start a fine-tuning job
- `.\.venv\Scripts\python backend/scripts/fine_tune.py start --training-jsonl backend/finetune/out/fastapi_docs_train_v1.jsonl --suffix fastapi_docs --n-epochs auto`

Note: if your base model is `gpt-4o-mini`, the script will automatically use a pinned snapshot (e.g. `gpt-4o-mini-2024-07-18`) because the generic alias may not be fine-tunable.

You can poll a job:
- `.\.venv\Scripts\python backend/scripts/fine_tune.py status --job-id <job_id>`

### 4) Enable the pipeline
Set one of:
- Recommended (per-domain): `finetuned_model: ft:...` in `domains/<domain>/config.yaml`
- Fallback (global): `OPENAI_FINETUNED_MODEL=ft:...`

Then run with pipelines including `finetune`:
- Frontend UI toggle
- Regression runner: `--pipelines prompt,rag,finetune`
