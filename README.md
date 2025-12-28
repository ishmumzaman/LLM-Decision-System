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
- LLM provider (MVP): OpenAI (keep code abstractable later).
- Python deps: venv + `requirements.txt`.
- Ship prebuilt RAG artifacts and commit them under `domains/<domain>/artifacts/`.
- Fairness: prompt-only and RAG share the same model + generation settings per `/run`.

## Next Steps (Backend)
1. Install deps: `python -m venv .venv` then `.\.venv\Scripts\python -m pip install -r backend/requirements.txt`
2. Set env: copy `backend/.env.example` -> `backend/.env` and set `OPENAI_API_KEY=...`
3. Fetch corpus (one-time): `python backend/scripts/fetch_fastapi_docs.py` (optional: use `--commit <sha>` for exact reproducibility)
4. Build artifacts (one-time): `python backend/scripts/build_index.py --domain fastapi_docs`
5. Run API: `.\.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --reload`
