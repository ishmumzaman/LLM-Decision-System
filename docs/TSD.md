# Technical Specification Document (TSD)
## Domain-Swappable LLM Decision System
### Comparing Prompt-Only vs RAG vs Fine-Tuned (Optional) Under Constraints

---

## Table of Contents
1. Purpose
2. System Summary
3. Goals and Non-Goals (Technical)
4. Assumptions and Constraints
5. Tech Stack
6. Repository and Directory Structure
7. Architecture Overview
8. Domain Abstraction Layer
9. Data Contracts (Schemas)
10. Backend Components
11. Pipeline Implementations
12. RAG Indexing and Retrieval
13. Evaluation Engine
14. Logging, Metrics, and Storage
15. API Specification
16. Frontend Specification
17. Performance, Reliability, and Failure Handling
18. Security Considerations
19. Testing Strategy
20. Deployment Specification
21. Operational Runbook (Minimal)
22. Future Extensions

---

## 1. Purpose

This TSD defines **how** the system is built and operated. It is intended to be:
- implementable as-is
- reviewable in an engineering design review
- defensible in interviews (clear component boundaries, explicit tradeoffs)

This system is not "just a chatbot." It is a **measurement + comparison platform** for LLM approaches.

---

## 2. System Summary

### 2.1 What the system does

Given:
- a user query
- a selected domain (e.g., developer documentation)
- a set of pipelines to run (prompt-only, RAG, optional fine-tuned)

The system:
1. loads the chosen domain configuration and artifacts
2. runs the selected pipelines **concurrently** under identical constraints
3. collects structured outputs (answer + metadata)
4. evaluates outputs with domain-aware heuristics
5. returns side-by-side results and metrics via API + UI

### 2.2 Primary outputs (what the system returns)

Per pipeline:
- answer text
- latency (ms)
- token usage (in/out when available)
- estimated cost (USD, when available)
- retrieved evidence (RAG only)
- evaluation flags (hallucination/grounding)
- heuristic quality score

Additionally:
- a comparison summary ("tradeoffs observed")

### 2.3 Core design principles

- **Domain-swappable**: new domains are added via config + data + rules, not core code edits
- **Pipeline-unified**: all pipelines produce a standardized result schema
- **Evaluation-first**: the product is the measured tradeoffs, not the raw text answer
- **Async by default**: pipeline runs are concurrent for fair latency comparison and better UX
- **Honest heuristics**: evaluation is explicit, explainable, and documented as approximate

---

## 3. Goals and Non-Goals (Technical)

### 3.1 Technical goals

- Provide a consistent orchestration layer for multiple pipelines
- Provide a consistent evaluation layer for multiple domains
- Provide a domain plug-in system that does not require rewriting pipeline code
- Provide reproducible experiment logs that can generate tables/plots for README
- Provide an interactive UI that makes tradeoffs visible (quality vs latency vs cost)

### 3.2 Technical non-goals

- Training large LLMs from scratch
- Perfect automated grading
- Production-scale multi-tenant auth and billing
- Handling sensitive/regulated workloads (medical/legal)
- SOTA benchmark chasing (this is systems + evaluation, not leaderboard work)

### 3.3 MVP Acceptance Criteria ("Definition of Done")

- [ ] `POST /run` runs `prompt` and `rag` concurrently and returns a valid `RunResponse`.
- [ ] `RunResponse` contains: `run_id`, `timestamp`, `domain`, `query`, `results`, `evaluations`, `summary_metrics`.
- [ ] `PipelineResult` contains (at minimum): `answer`, `latency_ms`, `tokens_in`, `tokens_out`, `cost_estimate_usd`, `retrieved_chunks`, `error`.
- [ ] RAG returns `retrieved_chunks` suitable for an evidence panel; prompt-only returns `retrieved_chunks = null`.
- [ ] Pipeline failures are isolated: one pipeline can error/timeout while others still return successfully.
- [ ] A fixed regression query set exists (20-50 queries) and is used before each release.
- [ ] UI renders side-by-side answers + per-pipeline metrics (latency/tokens/cost) and supports an expandable RAG evidence panel.
- [ ] Backend logs each run (JSONL or SQLite) with enough metadata to reproduce comparisons.
- [ ] Backend and frontend are deployed and the full flow works end-to-end from the UI.

---

## 4. Assumptions and Constraints

### 4.1 Assumptions

- LLM generation comes from **pre-trained models** (API or local open-source)
- RAG runs on a domain document corpus curated by the user
- Evaluation is heuristic; manual inspection can be used for calibration

### 4.2 Constraints

- MVP prioritizes: Prompt-only + RAG (fine-tuning is optional)
- "Domain swap" should be possible without touching core orchestration code
- Deployment is single-node (demo scale)

---

## 5. Tech Stack

### 5.1 Backend

- Python 3.11
- FastAPI (ASGI web service)
- Uvicorn (server)
- Pydantic (schemas + validation)
- Asyncio (concurrent pipeline runs)
- PyYAML (domain config loading)
- python-dotenv (local env management)
- Docker (container deployment)

### 5.2 ML / LLM

- Pre-trained LLM provider (API-based generation recommended for MVP)
- sentence-transformers (embedding model for RAG)
- FAISS (vector similarity search index)

### 5.3 Frontend

- React (minimal UI)
- Fetch/Axios (API calls)

### 5.4 Storage

- FAISS index file(s): `index.faiss`
- Chunk metadata file(s): `chunks.json`
- Run logs: JSONL (default) or SQLite (optional upgrade)

---

## 6. Repository and Directory Structure

Recommended structure:

```text
repo/
  backend/
    app/
      main.py
      settings.py
      schemas/
      domains/
      pipelines/
      rag/
      eval/
      logging/
    scripts/
      build_index.py
    tests/
  frontend/
    src/
    public/
  domains/
    registry.yaml
    fastapi_docs/
      config.yaml
      data/
      SOURCES.md
      artifacts/
        index.faiss
        chunks.json
  docs/
    PRD.md
    TSD.md
    EVALUATION.md
    ARCHITECTURE.md
```

Notes:
- Domain artifacts (FAISS + chunks metadata) are stored under `domains/<name>/artifacts/`.
- The backend should treat artifacts as read-only at runtime (unless rebuilding).

---

## 7. Architecture Overview

### 7.1 High-level flow

1. User submits query via UI.
2. UI sends request to `POST /run`.
3. Backend:
   - validates request
   - loads domain spec and artifacts
   - runs selected pipelines concurrently
   - evaluates results
   - logs the run
4. Backend returns structured response.
5. UI displays side-by-side answers + metrics + evidence.

### 7.2 Architecture diagram (logical)

```text
[React UI]
   |
   v
[FastAPI API Layer]
   |
   v
[Run Orchestrator]  (async)
   |
   +--> [Prompt-only Pipeline] ----+
   |                               |
   +--> [RAG Pipeline] ----------- +--> [Evaluation Engine] --> [Logs/Metrics]
   |
   +--> [Fine-tuned Pipeline]* ----+
          *optional
```

---

## 8. Domain Abstraction Layer

### 8.1 Why this exists

Different domains require different:
- corpora
- chunking strategies
- retrieval settings
- evaluation rules

We want to swap domains without rewriting core pipeline code.

### 8.2 Domain responsibilities

A domain provides:
- `documents_path`: corpus location
- chunk parameters: `chunk_size`, `chunk_overlap`
- retrieval parameter: `retrieval_k`
- `evaluation_rules`: list of rule identifiers
- optional `domain_prompt_prefix`

### 8.3 Domain config specification

Example `domains/fastapi_docs/config.yaml`:

```yaml
name: fastapi_docs
documents_path: ./domains/fastapi_docs/data
chunk_size: 500
chunk_overlap: 100
retrieval_k: 5
evaluation_rules:
  - invalid_api_check
  - unsupported_parameter_check
domain_prompt_prefix: |
  You are answering questions about FastAPI documentation.
  Prefer authoritative explanations grounded in the docs.
allowed_file_types:
  - md
  - txt
max_docs: 200
```

Required fields:
- `name`
- `documents_path`
- `chunk_size`
- `chunk_overlap`
- `retrieval_k`
- `evaluation_rules`

Optional fields:
- `domain_prompt_prefix`
- `allowed_file_types`
- `max_docs`

### 8.4 Domain registry

`domains/registry.yaml` lists which domains exist:

```yaml
domains:
  - fastapi_docs
  - react_docs
```

The registry enables:
- backend domain validation
- UI domain dropdown population
- domain discovery without code changes

### 8.5 Runtime `DomainSpec` object

At runtime, the backend constructs a `DomainSpec` object from registry + config:
- `name`
- `documents_path`
- `chunk_size`
- `chunk_overlap`
- `retrieval_k`
- `evaluation_rules`
- `domain_prompt_prefix` (optional)
- artifact paths (`index.faiss`, `chunks.json`)

This object is passed into:
- pipelines
- evaluation engine

---

## 9. Data Contracts (Schemas)

### 9.1 Schema design goals

- Make pipeline outputs comparable (same fields across pipelines).
- Make failures non-fatal (errors captured per pipeline).
- Keep response stable as new pipelines/domains are added.

### 9.2 `RunRequest` (client -> backend)

Fields:
- `query` (string, required)
- `domain` (string, required)
- `pipelines` (string[], optional; default = `["prompt", "rag"]`)
- `run_id` (string, optional; generated if absent)
- `client_metadata` (object, optional; UI version, etc.)

Validation rules:
- `domain` must exist in registry
- `pipelines` must be subset of supported pipelines
- `query` length capped (basic abuse prevention)

### 9.3 `PipelineResult` (per pipeline)

Fields:
- `pipeline` (string)
- `model` (string; provider model id used for generation)
- `generation_config` (object; e.g., `temperature`, `top_p`, `max_tokens`, `seed` when supported)
- `answer` (string)
- `latency_ms` (int)
- `tokens_in` (int | null)
- `tokens_out` (int | null)
- `cost_estimate_usd` (float | null)
- `retrieved_chunks` (`RetrievedChunk[]` | null)
- `flags` (object/dict; reserved for runtime notes)
- `error` (string | null)

### 9.4 `RetrievedChunk` (RAG evidence)

Fields:
- `chunk_id` (string | int)
- `source` (string; filename or URL label)
- `text_preview` (string; truncated for UI)
- `score` (float | null; similarity score)

### 9.5 `EvaluationResult` (per pipeline)

Fields:
- `quality_score` (float 0..1 | null)
- `hallucination_flags` (string[])
- `grounding_flags` (string[])
- `notes` (string | null)

### 9.6 `RunResponse` (backend -> client)

Fields:
- `run_id`
- `timestamp` (ISO-8601)
- `domain`
- `query`
- `results` (map: pipeline -> `PipelineResult`)
- `evaluations` (map: pipeline -> `EvaluationResult`)
- `summary_metrics` (object; comparison summary)

---

## 10. Backend Components

### 10.1 API Layer (FastAPI)

Responsibilities:
- validate request schema
- domain registry discovery
- orchestration of pipelines
- return response schema

Endpoints:
- `GET /health`
- `GET /domains`
- `POST /run`

### 10.2 Settings and configuration

The backend loads:
- environment variables (LLM provider key, model name, default settings)
- domain registry location
- artifact root directory
- pipeline timeout settings

### 10.3 Domain loader

Responsibilities:
- read `domains/registry.yaml`
- validate domain exists
- load `domains/<domain>/config.yaml`
- return `DomainSpec` with resolved paths
- validate artifacts exist (for RAG) or provide actionable error

### 10.4 Run orchestrator (pipeline router)

Responsibilities:
- take `RunRequest` + `DomainSpec`
- create a single `GenerationConfig` for the run (shared across pipelines)
- run selected pipelines concurrently
- enforce timeouts per pipeline
- collect `PipelineResult` objects
- call evaluation engine
- persist logs
- return `RunResponse`

Concurrency requirement:
- pipelines must run concurrently to avoid sequential latency stacking

Fairness requirement:
- within a single `/run`, all pipelines must use the same base model and generation settings (`GenerationConfig`); only the retrieval/context step is allowed to differ

Partial failure behavior:
- one pipeline failing must not break the entire response
- return results for pipelines that succeeded
- include `error` field for failures

---

## 11. Pipeline Implementations

### 11.1 Shared pipeline requirements

All pipelines must:
- accept the same inputs (`query` + `DomainSpec` + settings)
- use the same `GenerationConfig` for a given run (no per-pipeline temperature/max token overrides)
- return `PipelineResult` (same schema)
- include latency and error info

### 11.2 Prompt-only pipeline

Purpose:
- baseline for quality/hallucination/latency/cost

Flow:
1. Build prompt:
   - system instruction
   - optional `domain_prompt_prefix`
   - user query
2. Call LLM provider.
3. Measure latency.
4. Capture token usage if available.
5. Return `PipelineResult`.

Key design choice:
- keep prompt template stable to ensure comparable runs

### 11.3 RAG pipeline

Purpose:
- evaluate grounding gains vs added latency/cost/complexity

Online flow:
1. Embed query.
2. Retrieve top-k chunks via FAISS.
3. Assemble context:
   - include chunk text (bounded to context window)
   - include instruction: "use only provided context"
4. Call LLM provider.
5. Measure latency + token usage.
6. Return `PipelineResult` with retrieved evidence included.

Important:
- retrieved evidence is returned so UI can display "why" the answer was grounded

### 11.4 Fine-tuned pipeline (optional)

Purpose:
- quantify marginal benefit over RAG and baseline

Scope constraints:
- no training from scratch
- small-scale fine-tuning only
- must be justified via measured improvement vs cost/effort

Not required for MVP.

---

## 12. RAG Indexing and Retrieval

### 12.1 Indexing philosophy

Indexing should be:
- offline (precomputed) for speed and reproducibility
- artifact-based (index file + metadata)
- MVP decision: build offline and commit artifacts to the repo (do not build the index on service startup)

Polish / later:
- bake artifacts into the backend Docker image during the build (and stop committing artifacts if desired)

### 12.2 Index artifacts

For each domain:
- `index.faiss` (vector index)
- `chunks.json` (metadata mapping row -> chunk text/source)
- `index_meta.json` (embedding model id, chunk params, corpus hash, build timestamp)

### 12.3 Document ingestion

Ingestion rules:
- load only allowed file types
- cap total docs if `max_docs` is set
- preserve basic metadata (filename, section if detectable)

### 12.4 Chunking strategy

Chunking parameters:
- `chunk_size` (characters or approx tokens)
- `chunk_overlap`

Chunk metadata includes:
- `chunk_id`
- source filename
- character offsets (optional)
- chunk text

### 12.5 Embedding strategy

- embed all chunks using a single embedding model
- embed user query at runtime using same model
- embeddings must be consistent across indexing and querying

### 12.6 Retrieval strategy

- compute query embedding
- FAISS similarity search
- return top-k chunk ids
- map ids to chunk metadata
- send chunk texts into the prompt context

### 12.7 Retrieval failure modes

Empty retrieval results:
- return pipeline result with warning flag (e.g. `NO_RETRIEVAL_HITS`)

Irrelevant retrieval:
- evaluation may flag `NOT_GROUNDED_IN_CONTEXT` if answer conflicts with evidence

---

## 13. Evaluation Engine

### 13.1 Goals

Evaluation must:
- be consistent across pipelines
- be explainable and documented
- expose failure modes (hallucination/unsupported claims)
- enable tradeoff reasoning (quality vs latency vs cost)

### 13.2 Evaluation philosophy

- heuristic evaluation is acceptable and expected for MVP
- the system must be honest about limitations
- the goal is to compare pipelines fairly, not to claim absolute correctness

### 13.3 Metrics

Performance metrics (direct):
- `latency_ms`
- `tokens_in` / `tokens_out`
- `cost_estimate_usd`

Quality metrics (heuristic):
- `quality_score` (0..1)
- `hallucination_flags`
- `grounding_flags`

### 13.4 Domain-specific rule injection

Domain config lists `evaluation_rules`. The evaluation engine:
- loads rule functions by name
- applies each to each pipeline output
- aggregates flags and penalties

Examples for developer-doc domains:
- `invalid_api_check`: flags references to APIs not present in retrieved context / corpus markers
- `unsupported_parameter_check`: flags parameter names not supported by evidence
- `not_grounded_check`: for RAG, flags claims not supported by retrieved chunks
- `overconfident_language_check`: flags "guaranteed/always" style language if not grounded

### 13.5 Scoring strategy (simple and explainable)

Recommended approach:
- start with a base score (e.g. 1.0)
- subtract penalties for each flag category
- clamp to 0..1
- optionally add small positive credit for evidence usage (RAG)

This yields a consistent heuristic quality score.

### 13.6 Output format

Per pipeline:
- flags arrays (strings)
- quality score (float)
- short notes (human-readable summary)

### 13.7 Comparison summary generation

The evaluation layer also produces:
- `winner_by_quality`
- `winner_by_latency`
- `winner_by_cost`
- `tradeoff_summary` (one paragraph)

The system should never claim "best overall" without noting tradeoffs.

---

## 14. Logging, Metrics, and Storage

### 14.1 Why logging exists

Logging supports:
- reproducibility
- debugging
- offline analysis (generate plots/tables for README)
- progress tracking during development

### 14.2 What gets logged per run

- `timestamp`
- `run_id`
- `domain`
- `query`
- selected pipelines
- per-pipeline latency/tokens/cost
- per-pipeline model + generation config
- per-pipeline flags + score
- errors (if any)

### 14.3 Storage formats

MVP default:
- JSONL append-only log file (easy to inspect)

Optional upgrade:
- SQLite for easier querying

### 14.4 Privacy and safety

- do not store user PII
- sanitize or cap query lengths if public

---

## 15. API Specification

### 15.1 `GET /health`

Purpose:
- liveness check for deployment and monitoring

Response:
- simple `OK` or minimal JSON status

### 15.2 `GET /domains`

Purpose:
- list domains from registry

Response:
- array of domain names

### 15.3 `POST /run`

Purpose:
- run selected pipelines and return results + evaluations

Request body example:

```json
{
  "query": "How do I add middleware in FastAPI?",
  "domain": "fastapi_docs",
  "pipelines": ["prompt", "rag"]
}
```

Response body (high-level) example:

```json
{
  "run_id": "uuid",
  "timestamp": "2026-01-01T12:00:00Z",
  "domain": "fastapi_docs",
  "query": "How do I add middleware in FastAPI?",
  "results": {
    "prompt": {
      "pipeline": "prompt",
      "model": "provider/model-id",
      "generation_config": { "temperature": 0.2, "top_p": 1.0, "max_tokens": 800 },
      "answer": "...",
      "latency_ms": 650,
      "tokens_in": 900,
      "tokens_out": 400,
      "cost_estimate_usd": 0.01,
      "retrieved_chunks": null,
      "flags": {},
      "error": null
    },
    "rag": {
      "pipeline": "rag",
      "model": "provider/model-id",
      "generation_config": { "temperature": 0.2, "top_p": 1.0, "max_tokens": 800 },
      "answer": "...",
      "latency_ms": 1200,
      "tokens_in": 1400,
      "tokens_out": 420,
      "cost_estimate_usd": 0.03,
      "retrieved_chunks": [
        { "chunk_id": 12, "source": "docs/middleware.md", "text_preview": "...", "score": 0.82 }
      ],
      "flags": {},
      "error": null
    }
  },
  "evaluations": {
    "prompt": {
      "quality_score": 0.45,
      "hallucination_flags": ["INVALID_API_REFERENCE"],
      "grounding_flags": [],
      "notes": "Referenced an API not found in docs."
    },
    "rag": {
      "quality_score": 0.78,
      "hallucination_flags": [],
      "grounding_flags": [],
      "notes": "Grounded response with retrieved context."
    }
  },
  "summary_metrics": {
    "winner_by_quality": "rag",
    "winner_by_latency": "prompt",
    "winner_by_cost": "prompt",
    "tradeoff_summary": "RAG improved grounding at higher latency and cost."
  }
}
```

---

## 16. Frontend Specification

### 16.1 Responsibilities

The frontend exists to make comparisons legible:
- accept query
- allow domain selection
- allow pipeline selection
- show answers side-by-side
- show metrics and warnings
- show retrieved evidence for RAG

### 16.2 Minimum UI components

- Domain selector dropdown
- Pipeline toggle group (prompt / rag / finetune optional)
- Query input box
- Run button
- Results comparison view:
  - answer card per pipeline
  - metrics chips (latency/tokens/cost)
  - flags/warnings panel
  - evidence panel (RAG chunks, expandable)

### 16.3 UX requirements

- loading state while pipelines run
- error display per pipeline (don't hide failures)
- disable run button while in-flight
- show "partial results" if a pipeline fails

### 16.4 Non-goals for frontend

- auth
- persistent user accounts
- heavy styling/animations

---

## 17. Performance, Reliability, and Failure Handling

### 17.1 Performance targets (MVP-level)

- prompt-only should be fastest
- RAG should be slower but more grounded
- overall request time should be bounded by pipeline timeouts

### 17.2 Reliability requirements

- pipeline isolation: one failing pipeline does not fail the entire run
- timeouts enforced per pipeline
- backend returns structured errors, not stack traces

### 17.3 Common failure modes and behavior

LLM provider timeout:
- mark pipeline result `error = "TIMEOUT"`
- still return other pipelines

Missing FAISS artifacts:
- return actionable error recommending index build step

Invalid domain:
- return 400 with domain list

Empty retrieval results:
- RAG returns answer with warning flag (evaluation reflects lower confidence)

---

## 18. Security Considerations

This is a demo system, but basic hygiene still applies:
- API keys only via environment variables
- do not log secrets
- cap query length to avoid abuse/cost explosions
- optional rate limiting if publicly exposed
- do not store PII intentionally

---

## 19. Testing Strategy

### 19.1 Unit tests

- domain config loading + validation
- chunking behavior (size, overlap, deterministic output)
- FAISS retrieval returns top-k and stable mapping to chunks
- evaluation rules produce expected flags
- schema validation rejects invalid inputs

### 19.2 Integration tests

- `/run` returns valid schema for prompt + rag
- concurrent execution works (no sequential stacking)
- partial pipeline failure returns partial results correctly
- domain swap works by config change (no code edits)

### 19.3 Manual test suite (recommended)

Maintain a small query set (20-50) in a versioned file (e.g., `backend/tests/fixtures/mvp_queries.yaml`) and treat it as a regression suite:
- easy questions
- ambiguous questions
- hallucination traps
- `needle-in-haystack` doc queries
- retrieval failure cases

---

## 20. Deployment Specification

### 20.1 Backend deployment

- containerized FastAPI app
- environment variables set on platform
- ensure domain artifacts are available:
  - either baked into image
  - or mounted at runtime
  - (non-MVP fallback) built during startup (slower)
- MVP decision: build via `backend/scripts/build_index.py` and commit artifacts to the repo
- polish / later: bake artifacts into the backend Docker image during the build
- when committing artifacts, ensure the Docker build context includes `domains/**/artifacts/` (do not exclude it in `.dockerignore`)

### 20.2 Frontend deployment

- static React build deployed to a static host
- backend URL configured via environment variable

### 20.3 Deployment verification checklist

- `/health` works
- `/domains` returns expected list
- `/run` works for prompt-only
- `/run` works for RAG and returns evidence chunks
- UI can query backend and render results

---

## 21. Operational Runbook (Minimal)

### 21.1 If RAG retrieval fails

- check that `index.faiss` and `chunks.json` exist for the domain
- rebuild index via indexing script
- redeploy or restart service if artifacts are newly added

### 21.2 If costs spike unexpectedly

- reduce max tokens
- add request caps
- add basic rate limiting
- add caching for repeated queries

### 21.3 If latency is too high

- reduce `retrieval_k`
- reduce context size
- enable concurrency limits
- cache retrieval results (optional)

---

## 22. Future Extensions

These are explicitly "after MVP":
- add fine-tuning pipeline and quantify improvement vs cost/effort
- add reranking for RAG (hybrid retrieval improvements)
- add second domain to demonstrate domain-swappability
- add robustness tests (prompt perturbations, paraphrases, adversarial queries)
- add offline batch evaluation and leaderboard/report generation
- add experiment dashboard (basic charts over JSONL/SQLite logs)
