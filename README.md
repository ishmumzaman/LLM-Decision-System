# LLM Decision System
Domain-swappable evaluation system to compare prompt-only vs doc-grounded RAG (fine-tuning optional) on identical inputs, reporting deterministic metrics, latency, and estimated cost.

## Quickstart (Docker)
Prereqs: Docker.

1. Clone:
   - `git clone <repo-url>`
   - `cd <repo>`
2. Create env file:
   - macOS/Linux: `cp .env.example .env`
   - Windows (PowerShell): `Copy-Item .env.example .env`
3. Run:
   - `docker compose up --build`

Open:
- UI: `http://127.0.0.1:5173`
- API: `http://127.0.0.1:8000`

### Demo mode (no API key)
If `OPENAI_API_KEY` is blank and `DEMO_MODE=1`, the backend runs in **Demo mode** (no API calls) and returns bundled sample outputs so reviewers can use the UI immediately.

To run live:
- Set `OPENAI_API_KEY=...` in `.env` (repo root) and restart.

## Local dev (no Docker)
Prereqs: Python 3.11+ (see `.python-version`), Node 20+ (see `.nvmrc`), npm.

- Windows: `.\dev.ps1`
  - If scripts are blocked: `Set-ExecutionPolicy -Scope Process Bypass -Force`
- macOS/Linux: `./dev.sh`

## Docs
- `docs/Goal.md`
- `docs/PRD.md`
- `docs/TSD.md`
- `docs/Timeline.md`
- `docs/ExampleTimeline.md`

## MVP Decisions (Locked)
- Domains: FastAPI (`fastapi_docs`), React (`react_docs`), PostgreSQL (`postgresql_docs`)
- LLM provider (MVP): OpenAI (keep code abstractable later)
- Python deps: venv + `requirements.txt`
- Ship prebuilt RAG artifacts and commit them under `domains/<domain>/artifacts/`
- Fairness: prompt-only and RAG share the same model + generation settings per `/run`

## Backend (manual setup)
1. Install deps: `python -m venv .venv` then `.\.venv\Scripts\python -m pip install -r backend/requirements.txt`
2. Set env (either works):
   - Recommended: copy `.env.example` -> `.env` and set `OPENAI_API_KEY=...`
   - Alternative: copy `backend/.env.example` -> `backend/.env`
3. Fetch corpus (one-time):
   - FastAPI: `python backend/scripts/fetch_fastapi_docs.py` (optional: use `--commit <sha>` for exact reproducibility)
   - React: `python backend/scripts/fetch_react_docs.py` (optional: `--include <subdir>` repeatable, or `--include-blog`)
   - PostgreSQL: `python backend/scripts/fetch_postgresql_docs.py` (optional: use `--commit <sha>` for exact reproducibility)
4. Build artifacts (one-time):
   - FastAPI: `python backend/scripts/build_index.py --domain fastapi_docs`
   - React: `python backend/scripts/build_index.py --domain react_docs`
   - PostgreSQL: `python backend/scripts/build_index.py --domain postgresql_docs`
5. Run API: `.\.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --reload`

## Regression Suite
- FastAPI suite: `backend/tests/fixtures/mvp_queries.yaml`
  - Runner (backend must be running): `.\.venv\Scripts\python backend/scripts/run_regression.py --base-url http://127.0.0.1:8000 --mode docs`
  - Include fine-tune: `--pipelines prompt,rag,finetune` (requires a configured fine-tuned model)
- React suite: `backend/tests/fixtures/react_docs_mvp_v1.yaml`
  - Runner: `.\.venv\Scripts\python backend/scripts/run_regression.py --suite backend/tests/fixtures/react_docs_mvp_v1.yaml --pipelines prompt,rag`
- PostgreSQL suite: `backend/tests/fixtures/postgresql_docs_mvp_v1.yaml`
  - Runner: `.\.venv\Scripts\python backend/scripts/run_regression.py --suite backend/tests/fixtures/postgresql_docs_mvp_v1.yaml --pipelines prompt,rag`

## Modes (Docs-grounded vs General)
The UI/API supports a per-request `mode`:
- `docs`: docs-grounded behavior (RAG uses only retrieved domain context and cites chunk ids)
- `general`: general-knowledge behavior (RAG may optionally use retrieved context; grounding/citation checks are disabled)

## Fine-tuning (Optional)
This enables a third pipeline: `finetune`.

1. (Optional) Generate a dataset from the domain chunks:
   - `.\.venv\Scripts\python backend/scripts/fine_tune.py generate --domain fastapi_docs --count 100 --out-yaml backend/finetune/datasets/fastapi_docs_train_gen.yaml`
2. Convert a dataset YAML to OpenAI JSONL:
   - `.\.venv\Scripts\python backend/scripts/fine_tune.py prepare --domain fastapi_docs --in-yaml backend/finetune/datasets/fastapi_docs_train_v1.yaml --out-jsonl backend/finetune/out/fastapi_docs_train_v1.jsonl`
3. Start a fine-tuning job (uploads the JSONL):
   - `.\.venv\Scripts\python backend/scripts/fine_tune.py start --training-jsonl backend/finetune/out/fastapi_docs_train_v1.jsonl --suffix fastapi_docs --n-epochs auto`
     - Note: if your base model is `gpt-4o-mini`, the script will automatically use a pinned snapshot (e.g. `gpt-4o-mini-2024-07-18`) because the generic alias may not be fine-tunable.
4. When the job succeeds, set the returned fine-tuned model id:
   - Recommended (per-domain): set `finetuned_model: ft:...` in `domains/<domain>/config.yaml`
   - Fallback (global): `OPENAI_FINETUNED_MODEL=ft:...`
5. Run comparisons:
   - UI: toggle "Fine-tuned"
   - Regression: `.\.venv\Scripts\python backend/scripts/run_regression.py --pipelines prompt,rag,finetune`

## Frontend (MVP UI)
1. Install deps: `cd frontend` then `npm install`
2. Configure API URL (optional): copy `frontend/.env.example` -> `frontend/.env` and set `VITE_API_BASE_URL=http://127.0.0.1:8000`
3. Run dev server: `npm run dev`

## Docker (Backend + Frontend)
Local run (requires Docker):
1. Copy `.env.example` -> `.env` and set `OPENAI_API_KEY=...` (or leave blank to use Demo mode)
2. Start: `docker compose up --build`
3. Open UI: `http://127.0.0.1:5173` (API at `http://127.0.0.1:8000`)

