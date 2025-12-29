# LLM Decision System
Domain-swappable evaluation system to compare prompt-only vs RAG (fine-tuning optional) on identical inputs, reporting quality/grounding heuristics, latency, and cost.

## Docs
- `docs/Goal.md`
- `docs/PRD.md`
- `docs/TSD.md`
- `docs/Timeline.md`
- `docs/ExampleTimeline.md`

## MVP Decisions (Locked)
- Domain: FastAPI docs (`fastapi_docs`).
- Additional example domain (post-MVP): React docs (`react_docs`).
- LLM provider (MVP): OpenAI (keep code abstractable later).
- Python deps: venv + `requirements.txt`.
- Ship prebuilt RAG artifacts and commit them under `domains/<domain>/artifacts/`.
- Fairness: prompt-only and RAG share the same model + generation settings per `/run`.

## Next Steps (Backend)
1. Install deps: `python -m venv .venv` then `.\.venv\Scripts\python -m pip install -r backend/requirements.txt`
2. Set env: copy `backend/.env.example` -> `backend/.env` and set `OPENAI_API_KEY=...`
3. Fetch corpus (one-time):
   - FastAPI: `python backend/scripts/fetch_fastapi_docs.py` (optional: use `--commit <sha>` for exact reproducibility)
   - React: `python backend/scripts/fetch_react_docs.py` (optional: `--include <subdir>` repeatable, or `--include-blog`)
4. Build artifacts (one-time):
   - FastAPI: `python backend/scripts/build_index.py --domain fastapi_docs`
   - React: `python backend/scripts/build_index.py --domain react_docs`
5. Run API: `.\.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --reload`

## Regression Suite
- Query set: `backend/tests/fixtures/mvp_queries.yaml`
- Runner (backend must be running): `.\.venv\Scripts\python backend/scripts/run_regression.py --base-url http://127.0.0.1:8000`
  - Include fine-tune: `--pipelines prompt,rag,finetune` (requires a configured fine-tuned model)
- React suite: `backend/tests/fixtures/react_docs_mvp_v1.yaml`
  - Runner: `.\.venv\Scripts\python backend/scripts/run_regression.py --suite backend/tests/fixtures/react_docs_mvp_v1.yaml --pipelines prompt,rag`

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
   - UI: toggle “Fine-tuned”
   - Regression: `.\.venv\Scripts\python backend/scripts/run_regression.py --pipelines prompt,rag,finetune`

## Frontend (MVP UI)
1. Install deps: `cd frontend` then `npm install`
2. Configure API URL (optional): copy `frontend/.env.example` -> `frontend/.env` and set `VITE_API_BASE_URL=http://127.0.0.1:8000`
3. Run dev server: `npm run dev`

## Docker (Backend + Frontend)
Local run (requires Docker):
1. Set env var: `setx OPENAI_API_KEY "..."` (new terminal after) or export it in your shell
2. Start: `docker compose up --build`
3. Open UI: `http://127.0.0.1:5173` (API at `http://127.0.0.1:8000`)
