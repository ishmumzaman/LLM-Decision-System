# Example Timeline (Suggested Schedule)

This is a sample calendar-style plan that maps the checklist phases in `Timeline.md` onto a practical schedule. Adjust based on your available time.

---

## Week 0 (1-2 sessions): Lock decisions
- Lock MVP definition of done (PRD + TSD acceptance criteria).
- Pick initial domain corpus (10-50 pages) and create `SOURCES.md`.
- Choose the base generation model + generation config for fairness.
- Confirm MVP choice: build RAG artifacts offline and commit them to the repo.
- Start a regression query file (target 20-50 by the end).

---

## Week 1: Backend foundations + RAG indexing
- Scaffold repo structure + FastAPI skeleton (`/health`, `/domains`, `/run` stub).
- Implement domain loading (registry + config).
- Implement ingestion + chunking.
- Implement indexing script and produce `index.faiss`, `chunks.json`, `index_meta.json`.
- Sanity-check retrieval quality and iterate chunking/k early.

---

## Week 2: Pipelines + orchestration + evaluation
- Implement provider adapter (`generate()` + token/cost metadata when available).
- Implement prompt-only pipeline.
- Implement RAG pipeline (retrieve -> assemble context -> generate).
- Implement orchestrator with concurrency + per-pipeline timeouts + partial failure handling.
- Implement evaluation heuristics and summary metrics.
- Add JSONL logging for each run.

---

## Week 3: Frontend + end-to-end
- Build the minimal React UI (domain select, pipeline toggles, query input, run button).
- Render side-by-side results, metrics chips, and an expandable evidence panel for RAG.
- Add loading/disabled states and per-pipeline errors (partial results).
- Run the regression query set; tune prompts/retrieval/eval conservatively.

---

## Week 4: Deploy + polish
- Dockerize backend; confirm prebuilt artifacts are included in the image or mounted at runtime.
- Deploy backend + frontend and verify end-to-end.
- Update README with architecture + "how to add a domain" + a small results table.
- Record a short demo video.

Polish / later:
- Stop committing artifacts and bake them into the backend Docker image during the build.
