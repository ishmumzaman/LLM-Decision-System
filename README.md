# LLM Decision System
Domain-swappable evaluation harness to compare **prompt-only** vs **doc-grounded RAG** vs **(optional) fine-tuned** pipelines on identical inputs, with transparent metrics for grounding, latency, and cost.

This project is intentionally **evaluation-first**: the "product" is not a chatbot, it's a measurement + comparison platform that helps you make defensible architecture decisions (when is prompt-only enough? when is RAG worth it? when is fine-tuning justified?).

---

## Who this is for
- Students with basic AI/ML background who want a concrete, end-to-end "LLM systems" project (RAG + evaluation + UI + reproducibility).
- Senior AI/ML engineers and hiring managers who want to see good engineering judgment: fair comparisons, explicit tradeoffs, honest evaluation boundaries, and clean abstractions.
- And anyone in between this skill level

---

## Tech stack (what you will see in the code)
Backend:
- Python 3.11, FastAPI, Uvicorn
- Pydantic + pydantic-settings (schemas + config)
- asyncio (concurrent pipeline runs)
- OpenAI SDK (chat + embeddings)
- FAISS + NumPy (vector index + retrieval)

Frontend:
- React + TypeScript (Vite)
- Tailwind CSS
- Nginx (static hosting in Docker)

Ops / packaging:
- Dockerfiles for backend + frontend
- docker-compose for one-command local run
- demo mode + dev scripts for low-friction onboarding

---

## Table of contents
- [LLM Decision System](#llm-decision-system)
  - [Who this is for](#who-this-is-for)
  - [Tech stack (what you will see in the code)](#tech-stack-what-you-will-see-in-the-code)
  - [Table of contents](#table-of-contents)
  - [What the system does](#what-the-system-does)
  - [2-minute reviewer tour](#2-minute-reviewer-tour)
  - [UI walkthrough (what each control means)](#ui-walkthrough-what-each-control-means)
  - [Quickstart (Docker)](#quickstart-docker)
  - [Demo mode (no API key)](#demo-mode-no-api-key)
  - [Local dev (no Docker)](#local-dev-no-docker)
  - [Architecture overview](#architecture-overview)
    - [High-level system diagram](#high-level-system-diagram)
    - [Key design principles (from docs/PRD.md + docs/TSD.md)](#key-design-principles-from-docsprdmd--docstsdmd)
  - [Domains (swappable plug-ins)](#domains-swappable-plug-ins)
    - [Domain config schema (conceptual)](#domain-config-schema-conceptual)
    - [Adding a new domain (step-by-step)](#adding-a-new-domain-step-by-step)
  - [Pipelines](#pipelines)
    - [`prompt` (prompt-only baseline)](#prompt-prompt-only-baseline)
    - [`rag` (retrieval-augmented, evidence-citing)](#rag-retrieval-augmented-evidence-citing)
    - [`finetune` (optional per-domain)](#finetune-optional-per-domain)
  - [Modes: docs-grounded vs general](#modes-docs-grounded-vs-general)
    - [`docs` (docs-grounded)](#docs-docs-grounded)
    - [`general` (general knowledge)](#general-general-knowledge)
  - [Evaluation: metrics and scoring](#evaluation-metrics-and-scoring)
    - [Deterministic evaluation (stable, cheap)](#deterministic-evaluation-stable-cheap)
      - [1) Heuristic score (rules/penalties)](#1-heuristic-score-rulespenalties)
      - [2) Expectation score (deterministic pattern match)](#2-expectation-score-deterministic-pattern-match)
      - [3) Abstention correctness](#3-abstention-correctness)
      - [4) Latency + cost (telemetry)](#4-latency--cost-telemetry)
    - [LLM judge (optional; extra cost; non-deterministic)](#llm-judge-optional-extra-cost-non-deterministic)
    - [LLM proxies (optional; extra cost; non-deterministic)](#llm-proxies-optional-extra-cost-non-deterministic)
    - [Metrics Board + composite scoring (UI)](#metrics-board--composite-scoring-ui)
    - [Concrete "separating" effect (why the eval works)](#concrete-separating-effect-why-the-eval-works)
  - [Regression suites (reproducible comparisons)](#regression-suites-reproducible-comparisons)
    - [Runner](#runner)
  - [Fine-tuning (optional)](#fine-tuning-optional)
  - [Backend API reference](#backend-api-reference)
    - [`GET /health`](#get-health)
    - [`GET /domains`](#get-domains)
    - [`GET /suites` and `GET /suites/{id}`](#get-suites-and-get-suitesid)
    - [`POST /run`](#post-run)
  - [Env var reference](#env-var-reference)
    - [Required (for live runs)](#required-for-live-runs)
    - [Demo mode](#demo-mode)
    - [Provider + models](#provider--models)
    - [Generation config (shared across pipelines per run)](#generation-config-shared-across-pipelines-per-run)
    - [RAG caps](#rag-caps)
    - [Timeouts + safety](#timeouts--safety)
    - [Cost estimation (USD per 1M tokens)](#cost-estimation-usd-per-1m-tokens)
  - [Reproducibility and artifacts](#reproducibility-and-artifacts)
    - [What's committed vs not committed](#whats-committed-vs-not-committed)
    - [Rebuilding artifacts (offline)](#rebuilding-artifacts-offline)
  - [Deployment notes (when you're ready)](#deployment-notes-when-youre-ready)
  - [Security + safety notes](#security--safety-notes)
  - [Limitations (honest notes)](#limitations-honest-notes)
  - [Project docs (PRD/TSD)](#project-docs-prdtsd)

---

## What the system does
Given:
- a **domain** (e.g., FastAPI docs vs React docs vs PostgreSQL docs)
- a **mode** (docs-grounded vs general)
- a **query** (custom) or a **regression case** (pre-labeled)

The backend:
1. Runs selected pipelines **concurrently** under identical constraints:
   - `prompt` (prompt-only baseline)
   - `rag` (retrieval-augmented with evidence)
   - `finetune` (optional; per-domain)
2. Collects structured outputs per pipeline:
   - answer text
   - latency (ms)
   - token usage and estimated cost (if cost rates are configured)
   - retrieved chunks (RAG)
   - flags + errors (timeouts, missing artifacts, etc.)
3. Computes evaluation signals:
   - deterministic metrics (rule-based penalties, citations/grounding flags, expectation match, abstention correctness)
   - optional LLM judge (anonymized + pairwise rubric scoring)
   - optional LLM proxy signals (evidence support check, answerability estimation)
4. Returns a single `RunResponse` used by the UI and regression runner.

The frontend:
- displays pipeline outputs side-by-side
- shows raw metrics and flags
- provides a **Metrics Board** with:
  - stable presets (comparable), and
  - a fully configurable "User score (custom)" behind a hard warning + config hash

---

## 2-minute reviewer tour
If you're reviewing this repo (student, hiring manager, or engineer), this is the fastest way to understand what it does:

1. Run it in Demo mode (no API key): `docker compose up --build`
2. Open UI: `http://127.0.0.1:5173`
3. Switch **Mode** to "Docs-grounded" and run a query:
   - Observe: RAG shows evidence chunks; prompt-only is flagged as having no evidence source in docs mode.
4. Switch **Mode** to "General" and run a general question:
   - Observe: prompt-only may be enough; RAG costs more and can be slower.
5. Toggle **Judge** and/or **Evidence (LLM)** to see how non-deterministic eval signals work (with clear labeling that they cost $ and add variance).
6. Open the **Metrics Board**:
   - Pick a preset (stable, comparable).
   - Optionally enable custom scoring (hard warning + config hash).

That sequence demonstrates the core point: this is a **decision tool** for LLM architecture, not a single-answer "chatbot".

---

## UI walkthrough (what each control means)
This is the mental model for the UI:

- Domain: which corpus + artifacts are active (FastAPI / React / PostgreSQL in this repo).
- Mode:
  - Docs-grounded: "only answer if supported by this domain's corpus"
  - General: "answer normally; retrieval is optional context"
- Pipelines: which systems to run side-by-side (`prompt`, `rag`, `finetune` when available).
- Query source:
  - Custom: you type any query (no deterministic expect by default).
  - Regression case: you pick a suite + case with labels/tags/expect for reproducible evaluation.
- Optional eval toggles:
  - Judge: runs an extra rubric-based model call (costs $).
  - Evidence (LLM): checks if the retrieved chunks actually support the answer (costs $).
  - Answerability (LLM): estimates if the question needs docs (costs $).
- Metrics Board:
  - Presets: stable composite scoring (comparable across runs).
  - Advanced custom scoring: fully configurable; the UI warns about "false precision" and shows a config hash.

---

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

---

## Demo mode (no API key)
If `OPENAI_API_KEY` is blank and `DEMO_MODE=1`, the backend runs in **Demo mode**:
- no OpenAI API calls
- returns bundled sample outputs from `backend/demo_runs.json`
- the UI shows `Backend: demo`

To run live:
- set `OPENAI_API_KEY=...` in `.env` (repo root) and restart Docker Compose.

Important: Demo mode is intentionally "offline replay". The backend swaps your query to a bundled sample query so the UI is consistent without any external calls.

---

## Local dev (no Docker)
Prereqs: Python 3.11+ (see `.python-version`), Node 20+ (see `.nvmrc`), npm.

- Windows: `.\dev.ps1`
  - If scripts are blocked: `Set-ExecutionPolicy -Scope Process Bypass -Force`
- macOS/Linux: `./dev.sh` (may require `chmod +x dev.sh`)

Both scripts:
- create `.env` from `.env.example` if missing
- create a venv and install backend deps
- install frontend deps
- start backend (`uvicorn`) and frontend (`vite`)

---

## Architecture overview

### High-level system diagram
```mermaid
flowchart LR
  UI[React UI] -->|POST /run| API[FastAPI backend]

  API --> ORCH[Orchestrator]
  ORCH --> P1[prompt pipeline]
  ORCH --> P2[rag pipeline]
  ORCH --> P3[finetune pipeline (optional)]

  P2 --> RET[Retriever (FAISS)]
  RET --> ART[domains/<domain>/artifacts]

  ORCH --> EVAL[Deterministic evaluator]
  ORCH -->|optional| JUDGE[LLM judge (pairwise, anonymized)]
  ORCH -->|optional| PROXY[LLM proxies (evidence/answerability)]

  EVAL --> RESP[RunResponse JSON]
  JUDGE --> RESP
  PROXY --> RESP
  RESP --> UI
```

### Key design principles (from docs/PRD.md + docs/TSD.md)
- Domain-swappable: domains are config + data + rules (no core code edits).
- Pipeline-unified: all pipelines return the same schema (`PipelineResult`).
- Evaluation-first: show tradeoffs, not "vibes".
- Async by default: pipelines run concurrently for fair latency comparison.
- Honest heuristics: deterministic checks are explicit proxies, not "truth".

---

## Domains (swappable plug-ins)
Domains live under `domains/<domain>/` and are registered in `domains/registry.yaml`.

Each domain provides:
- corpus path (`documents_path`)
- chunking params (`chunk_size`, `chunk_overlap`)
- retrieval param (`retrieval_k`)
- evaluation rules to apply (`evaluation_rules`)
- optional domain prompt prefix (`domain_prompt_prefix`)
- optional fine-tuned model id (`finetuned_model`)

Example domains included:
- `fastapi_docs`
- `react_docs`
- `postgresql_docs`

### Domain config schema (conceptual)
Each `domains/<domain>/config.yaml` contains:
```yaml
name: <domain_name>
documents_path: ./domains/<domain>/data
chunk_size: 800
chunk_overlap: 120
retrieval_k: 6
evaluation_rules:
  - not_grounded_check
  - overconfident_language_check
domain_prompt_prefix: |
  Domain-specific system prompt prefix...
finetuned_model: ft:...   # optional
allowed_file_types: [md]  # or mdx, sgml, etc.
max_docs: 200
```

### Adding a new domain (step-by-step)
1. Create `domains/<new_domain>/config.yaml` and `domains/<new_domain>/SOURCES.md`.
2. Add `<new_domain>` to `domains/registry.yaml`.
3. Put your raw corpus under `domains/<new_domain>/data/` (gitignored).
4. Build RAG artifacts offline:
   - Windows: `.\.venv\Scripts\python backend/scripts/build_index.py --domain <new_domain>`
   - macOS/Linux: `.venv/bin/python backend/scripts/build_index.py --domain <new_domain>`
5. Add a regression suite YAML under `backend/tests/fixtures/` for coverage and reproducibility.

---

## Pipelines

### `prompt` (prompt-only baseline)
- Calls the base model (default `OPENAI_MODEL=gpt-4o-mini`).
- Uses the same generation settings as other pipelines.
- No retrieval; no evidence chunks.

### `rag` (retrieval-augmented, evidence-citing)
RAG relies on **offline artifacts** (committed for the MVP):
- `domains/<domain>/artifacts/index.faiss`
- `domains/<domain>/artifacts/chunks.json`
- `domains/<domain>/artifacts/index_meta.json`

Runtime behavior:
1. embed query (OpenAI embeddings; default `OPENAI_EMBEDDING_MODEL=text-embedding-3-small`)
2. retrieve top-k chunk ids from FAISS
3. build a context block within a budget (`RAG_MAX_CONTEXT_CHARS`)
4. generate answer
5. in docs mode, enforce citations like `[123]` (and rewrite once if missing)

Typical RAG failure modes are handled as structured errors:
- missing artifacts -> `MISSING_RAG_ARTIFACTS: build with ...`
- embedding model mismatch -> `RAG_ARTIFACT_MISMATCH`
- empty retrieval -> `NO_RETRIEVAL_HITS` flag and (docs mode) expected abstention

### `finetune` (optional per-domain)
- Calls a fine-tuned OpenAI model id.
- Config can be per-domain (`domains/<domain>/config.yaml`) or global (`OPENAI_FINETUNED_MODEL`).
- If missing for a domain, the pipeline returns a structured error and the regression runner treats it as **skipped**, not failed.

---

## Modes: docs-grounded vs general
Every run has a `mode`:

### `docs` (docs-grounded)
Goal: measure whether an approach can stay grounded in the selected domain corpus.

- RAG is instructed to answer using only retrieved context and cite chunk ids.
- Prompt-only has no retrieved context. To prevent misleading "prompt wins" outcomes in docs-grounded comparisons, the backend applies an explicit docs-grounded evidence rule:
  - if prompt-only does not abstain, it is flagged `PROMPT_NO_EVIDENCE_IN_DOCS` and penalized.

This is a deliberate design choice: docs-grounded mode is for "answer only if the docs support it."

### `general` (general knowledge)
Goal: measure normal usage of a base model.

- Prompt-only answers normally.
- RAG may optionally use retrieved context, but citations/grounding checks are not enforced.

Recommendation:
- Use `general` for broadly answerable questions.
- Use `docs` when you care about "only if supported by the corpus."

---

## Evaluation: metrics and scoring
This project intentionally avoids pretending there is a single ground-truth "quality" metric. Instead, it exposes multiple signals and clearly distinguishes:
- deterministic metrics (stable, comparable)
- LLM-based metrics (extra cost + variance; clearly labeled)

### Deterministic evaluation (stable, cheap)
Computed without additional model calls.

#### 1) Heuristic score (rules/penalties)
Implemented in `backend/app/eval/evaluator.py`.

Current deterministic rules include:
- `overconfident_language_check`:
  - flags "always/never/guaranteed/100%/must"
  - penalty: 0.15
- `not_grounded_check` (RAG):
  - in docs mode, requires citations like `[12]`
  - validates citations reference retrieved chunk ids
  - flags: `NO_EVIDENCE`, `MISSING_CITATIONS`, `INVALID_CITATIONS`
- `prompt_docs_evidence_check` (docs mode only; implemented in `backend/app/main.py`):
  - flags `PROMPT_NO_EVIDENCE_IN_DOCS`
  - penalizes prompt-only answers that do not abstain, because prompt has no evidence source in docs mode

The output includes:
- `quality_score = 1.0 - penalties`
- `rule_breakdown` explaining exactly which rule fired and why
- `hallucination_flags` and `grounding_flags` (explicit strings)

#### 2) Expectation score (deterministic pattern match)
Regression cases can include an `expect` block:
- `must_include` (all required)
- `must_include_any` (at least one from each group)
- `must_not_include` (forbidden)
- `expect_idk` (abstention required)

This provides a simple, reproducible "did it include the key idea?" check.

#### 3) Abstention correctness
Cases can label when abstention is correct (e.g., traps/unsupported claims) using:
- `expected_abstain_in_docs: true`

In docs mode, "I don't know" can be correct behavior and scored as such.

#### 4) Latency + cost (telemetry)
- latency is measured per pipeline
- cost is estimated from token usage and configured per-1M token rates (`OPENAI_COST_*`)

### LLM judge (optional; extra cost; non-deterministic)
Implemented in `backend/app/eval/judge.py`.

Why it exists:
- deterministic metrics can saturate
- a strict rubric-based judge can break ties if used carefully

Key properties:
- anonymized answers (A/B) to reduce style/preference bias
- pairwise comparisons (more stable than absolute 1-10 ratings)
- rubric criteria: correctness, groundedness, hallucination, abstention, usefulness, conciseness (0/1/2)
- fixed weights; temperature locked to 0
- in docs mode, includes RAG retrieved context previews when comparing against RAG
- UI allows choosing `judge_model` (recommended: different/stronger than the participant model)

### LLM proxies (optional; extra cost; non-deterministic)
Implemented in `backend/app/eval/proxies.py`.

1) Evidence support check (docs mode):
- asks: "are the answer's claims supported by retrieved context?"
- returns `support_score` in {0,1,2} and a list of unsupported claims

2) Answerability estimation:
- labels whether a question is answerable without reading docs:
  - `general`, `requires_docs`, `unsupported`, `unknown`
- helpful for custom queries and dataset curation

### Metrics Board + composite scoring (UI)
Implemented in `frontend/src/lib/scoring.ts` and `frontend/src/App.tsx`.

The UI always shows raw metrics, and optionally computes a composite score.

Presets (stable, comparable):
- Docs-grounded (preset)
- General (preset)
- Cost-sensitive (preset)
- Latency-sensitive (preset)

Custom composite scoring:
- behind an explicit warning + "I understand" confirmation
- labeled as `User score (custom)`
- config hash is displayed and stored with each run
- runs are only comparable if their config hashes match

Hard gates (optional):
- require full expectation match (expect=1.0)
- require abstention correctness (1.0)
- require no grounding flags
- require no hallucination flags

### Concrete "separating" effect (why the eval works)
This project evolved to avoid a common eval trap: prompt-only and RAG both scoring "1.0" on everything.

Two key upgrades make separation observable:
1) suite/case metadata (tags + requires_docs + expected_abstain labels)
2) stricter docs-grounded grounding logic (prompt-only penalized if it answers without evidence)

In a recent docs-mode sweep across the three example domains (62 cases total):
- prompt-only triggered `PROMPT_NO_EVIDENCE_IN_DOCS` in 59/62 cases (~95%)
- doc-grounded RAG triggered 0 of that flag (by design it has evidence)

Example delta-based scoring from the regression runner (docs mode, prompt vs rag):
- `fastapi_docs` (25 cases): delta_rag_vs_prompt_avg ~= 0.84
- `react_docs` (23 cases): delta_rag_vs_prompt_avg ~= 0.91
- `postgresql_docs` (14 cases): delta_rag_vs_prompt_avg ~= 0.71

These numbers are not "universal truth" (different models/prompts change them), but they show the harness is actually separating pipelines and surfacing tradeoffs.

---

## Regression suites (reproducible comparisons)
Regression suites are YAML files under `backend/tests/fixtures/`. Each suite declares:
- `suite` name, `domain`, optional `description`
- `queries`: list of cases

Each case can include:
- `id`: stable identifier
- `query`: text
- `tags`: for aggregation (e.g. `requires_docs`, `trap`, `edge_case`, `unsupported_claim`, `version_diff`)
- `expect`: deterministic expectation patterns
- labels: `requires_docs`, `answerable_from_general_knowledge`, `expected_abstain_in_docs`

### Runner
The runner calls the backend and prints per-case outcomes + aggregated stats, including delta scoring:

Delta scoring definition (RAG vs prompt):
- rank key = `(quality_score * expect_score, -#grounding_flags, -#hallucination_flags, -cost, -latency)`
- per case:
  - +1 if RAG rank > prompt rank
  - 0 if tie
  - -1 if prompt rank > RAG rank

Run it (backend must be running):
```bash
.\.venv\Scripts\python backend/scripts/run_regression.py --mode docs --pipelines prompt,rag
```

Suite selection:
```bash
.\.venv\Scripts\python backend/scripts/run_regression.py --suite backend/tests/fixtures/react_docs_mvp_v1.yaml --mode docs --pipelines prompt,rag
```

Write JSONL (useful for baselines; gitignored):
```bash
.\.venv\Scripts\python backend/scripts/run_regression.py --suite backend/tests/fixtures/mvp_queries.yaml --mode docs --pipelines prompt,rag,finetune --out-jsonl backend/regression_runs/fastapi_docs_full.jsonl
```

See all options:
```bash
.\.venv\Scripts\python backend/scripts/run_regression.py --help
```

---

## Fine-tuning (optional)
Fine-tuning is implemented as an optional third pipeline. It is per-domain by nature:
- domains have different vocabularies/APIs/failure modes
- you can add fine-tuning for a new domain without changing core orchestration/eval code

Repo workflow (see `docs/FINE_TUNING.md`):
1. create dataset YAML (manual or generated)
2. convert dataset YAML -> OpenAI JSONL
3. start fine-tuning job
4. set model id per-domain or globally

CLI entrypoints:
```bash
.\.venv\Scripts\python backend/scripts/fine_tune.py --help
```

---

## Backend API reference
Key endpoints (see `backend/app/main.py`):

### `GET /health`
Returns:
```json
{ "status": "ok", "demo_mode": true }
```

### `GET /domains`
Returns available domains and whether fine-tuning is configured per domain.

### `GET /suites` and `GET /suites/{id}`
Returns regression suite metadata and cases (used by the UI "Regression case" source).

### `POST /run`
Runs pipelines and returns a `RunResponse`.

Example (bash):
```bash
curl -s http://127.0.0.1:8000/run \
  -H 'Content-Type: application/json' \
  -d '{"domain":"fastapi_docs","mode":"docs","query":"What is FastAPI?","pipelines":["prompt","rag"]}'
```

Example (PowerShell):
```powershell
$body = @{
  domain = "fastapi_docs"
  mode = "docs"
  query = "What is FastAPI?"
  pipelines = @("prompt","rag")
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/run -ContentType "application/json" -Body $body
```

The response includes:
- `results[pipeline]`: answer, latency_ms, tokens_in/out, cost_estimate_usd, retrieved_chunks, flags, error
- `evaluations[pipeline]`: heuristic score, flags, rule breakdown, expect_score, abstention metrics
- `summary_metrics`: run-level metadata and winners
- optional `judge` and `proxies` blocks

---

## Env var reference
Backend settings are loaded from `backend/.env` and/or repo-root `.env` (both supported).

### Required (for live runs)
- `OPENAI_API_KEY`: OpenAI API key.

### Demo mode
- `DEMO_MODE` (backend default: `0`, this repo's `.env.example`: `1`): if true and no API key, backend serves bundled demo runs.
- `DEMO_RUNS_PATH` (default: `backend/demo_runs.json`): demo runs file.

### Provider + models
- `LLM_PROVIDER` (default: `openai`)
- `OPENAI_MODEL` (default: `gpt-4o-mini`)
- `OPENAI_JUDGE_MODEL` (default: `gpt-4o`)
- `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `OPENAI_FINETUNED_MODEL` (optional): global fine-tuned id

### Generation config (shared across pipelines per run)
- `GEN_TEMPERATURE` (default: `0.2`)
- `GEN_TOP_P` (default: `1.0`)
- `GEN_MAX_OUTPUT_TOKENS` (default: `800`)

### RAG caps
- `RAG_MAX_CONTEXT_CHARS` (default: `12000`)
- `RAG_MAX_CHUNKS` (default: `8`)

### Timeouts + safety
- `PIPELINE_TIMEOUT_S` (default: `30`)
- `MAX_RUN_COST_USD` (optional): best-effort cap; sets a `COST_CAP_EXCEEDED` flag when exceeded

### Cost estimation (USD per 1M tokens)
If unset, cost fields are null.
- `OPENAI_COST_INPUT_PER_1M`
- `OPENAI_COST_OUTPUT_PER_1M`
- `OPENAI_COST_INPUT_PER_1M_FINETUNED`
- `OPENAI_COST_OUTPUT_PER_1M_FINETUNED`

---

## Reproducibility and artifacts

### What's committed vs not committed
Committed (MVP choice):
- `domains/<domain>/artifacts/` (FAISS index + chunks + index meta)
- suite YAMLs under `backend/tests/fixtures/`

Not committed:
- `domains/<domain>/data/` (raw corpora; rebuild via fetch scripts)
- local run logs (e.g. `backend/runs.jsonl`, `backend/regression_runs/*.jsonl`)
- secrets (`.env`)

### Rebuilding artifacts (offline)
You do NOT need to rebuild artifacts to run the app (artifacts are committed for the MVP).

If you want to refresh or add a domain, rebuilding requires:
- `OPENAI_API_KEY` (embeddings)
- `git` (the fetch scripts clone upstream docs repos)

Fetch a corpus (optional; raw data is gitignored):
```bash
.\.venv\Scripts\python backend/scripts/fetch_fastapi_docs.py
```

Other fetchers:
```bash
.\.venv\Scripts\python backend/scripts/fetch_react_docs.py
.\.venv\Scripts\python backend/scripts/fetch_postgresql_docs.py
```

All fetch scripts support pinning an exact commit for reproducibility (see `--help`).

Build artifacts for a domain:
```bash
.\.venv\Scripts\python backend/scripts/build_index.py --domain fastapi_docs
```

MVP artifact strategy:
- artifacts are built offline and committed, so a reviewer can run the app immediately
- later/polish: bake artifacts into the backend Docker image (or mount them) instead of committing

---

## Deployment notes (when you're ready)
This repo is set up so local Docker is already "deployment-shaped":
- backend is a stateless FastAPI service (environment-driven config)
- frontend is a static build served via nginx

When deploying for real, the key requirements are:
- set `OPENAI_API_KEY` in the backend environment
- ensure `domains/<domain>/artifacts/` are present in the deployed backend (baked into image or mounted)
- point the frontend at the deployed backend base URL (`VITE_API_BASE_URL` at build time)

The detailed deployment checklist lives in `docs/TSD.md`.

---

## Security + safety notes
This is a demo-scale project (no auth, no multi-tenancy). Basic hygiene still applies:
- do not commit `.env` files or API keys
- consider adding rate limiting before exposing publicly
- use `MAX_RUN_COST_USD` to reduce runaway spend
- be aware the backend currently enables permissive CORS for local/dev convenience

---

## Limitations (honest notes)
- Deterministic metrics are not truth; they are proxies designed to surface specific failure modes (grounding, overconfidence, expectation match, abstention correctness).
- LLM judges/proxies can introduce bias and variance; this is why they are optional, temperature-0, and clearly labeled as non-deterministic.
- Docs-grounded mode intentionally enforces evidence. If you want "normal usage", use general mode.
- True correctness in open-ended QA often requires human evaluation; this system reduces guesswork, it doesn't eliminate it.

---

## Project docs (PRD/TSD)
For the full design spec and project planning artifacts:
- `docs/Goal.md` (motivation and goals)
- `docs/PRD.md` (product requirements)
- `docs/TSD.md` (technical specification)
- `docs/Timeline.md` and `docs/ExampleTimeline.md` (project plan)
- `docs/FINE_TUNING.md` (fine-tuning workflow)
