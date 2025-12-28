# Step-by-Step Objective List (Start -> Deployment)
## Domain-Swappable LLM Decision System (Prompt-only vs RAG vs Fine-tuning)

This is a complete, ordered checklist you can follow from **day 0** to a **fully deployed** system. It is written to prevent scope creep and make sure you always know what to do next.

---

## Phase 0 - Define Scope and "Done"

### 0.1 Lock the MVP (non-negotiable)
Decide what "done" means for your first shipped version:
- **One domain** (Developer Documentation)
- **Two pipelines**: Prompt-only + RAG
- **Metrics**: latency, token usage/cost estimate, hallucination flags, simple quality heuristic
- **UI**: side-by-side outputs + metrics display + domain/pipeline toggles
- **Deployment**: backend and frontend live
- **README**: architecture + how to add a domain + example results table

Do **not** include fine-tuning in MVP unless everything above is complete.

### 0.2 Choose a name and domain
Pick:
- A project name (can change later)
- A first domain (recommended: developer documentation)

### 0.3 Create success criteria
Write 5 test statements you can verify later, e.g.:
- "I can switch domains by changing config, without editing core logic."
- "Prompt-only and RAG run concurrently and return structured results."
- "UI shows answers side-by-side and displays latency + token usage."

### 0.4 Lock fairness rules (required for credible comparisons)
Define and freeze the invariants for MVP:
- same base generation model across pipelines per run
- same generation settings across pipelines (temperature/top_p/max_tokens/seed if supported)
- pipelines run concurrently under identical timeouts
- log generation settings with each run so comparisons are defensible

### 0.5 Choose the RAG index build strategy (MVP: commit artifacts to repo)
MVP decision:
- build artifacts offline via `backend/scripts/build_index.py`
- commit artifacts to the repo under `domains/<domain>/artifacts/`
- do not build the index during service startup for MVP
- ensure `.gitignore` and `.dockerignore` do not exclude `domains/**/artifacts/`

Polish / later:
- stop committing artifacts and bake them into the backend Docker image during the build

### 0.6 Start the regression query set early
Create a file (e.g., `backend/tests/fixtures/mvp_queries.yaml`) and start adding queries now.
You will expand this into the 20-50 query suite used before each release.

---

## Phase 1 - Repo Setup and Organization

### 1.1 Create the repository structure
Create these top-level folders:
- `backend/` (FastAPI + pipelines + evaluation)
- `frontend/` (React UI)
- `domains/` (domain plug-ins: data + config + eval rules)
- `docs/` (PRD, TSD, architecture diagrams, notes)

### 1.2 Add project documentation stubs
Create placeholder docs:
- `docs/PRD.md`
- `docs/TSD.md`
- `docs/ARCHITECTURE.md`
- `docs/EVALUATION.md`

### 1.3 Add standard repo hygiene
- `.gitignore`
- `LICENSE` (pick one early)
- `README.md` (basic skeleton + MVP checklist)
- Issue tracker or simple checklist in GitHub Projects (optional but helpful)

---

## Phase 2 - Domain Data Collection (Developer Docs)

### 2.1 Choose exactly what counts as the domain corpus
Decide what documents are authoritative for the domain. Examples:
- FastAPI docs pages
- Official reference docs for a library/framework
- A curated set of Markdown/HTML pages

Avoid mixing random blog posts at the start.

### 2.2 Collect the documents
Store them in:
- `domains/<domain_name>/data/`

Keep the corpus small at first:
- 10-50 pages worth of content is enough to start.

### 2.3 Track sources and attribution
Create:
- `domains/<domain_name>/SOURCES.md`
Include:
- links to each source
- date accessed
- any license/usage notes

---

## Phase 3 - Domain Plug-in Contract (Make It Swappable)

### 3.1 Define what a domain provides
A domain should provide:
- corpus location
- chunking parameters (size, overlap)
- retrieval parameters (top-k)
- evaluation rules (domain-specific heuristics)
- optional system prompt prefix (domain instruction)

### 3.2 Create a domain config file
Add:
- `domains/<domain_name>/config.yaml`

Include fields like:
- `name`
- `documents_path`
- `chunk_size`
- `chunk_overlap`
- `retrieval_k`
- `evaluation_rules` (rule names)

### 3.3 Create a domain registry
Add:
- `domains/registry.yaml`
List available domains and their config paths.

This enables domain listing in the UI and backend.

---

## Phase 4 - Backend Foundations (FastAPI Core)

### 4.1 Initialize backend environment
Set up:
- dependency manager (requirements or modern equivalent)
- environment variable handling (API keys, settings)
- local run instructions

### 4.2 Create the FastAPI app skeleton
Add:
- health endpoint
- domain listing endpoint
- placeholder run endpoint

### 4.3 Define strict request/response schemas
Design schemas for:
- request payload: query, domain, pipelines selected
- pipeline outputs: answer, latency, tokens, cost, retrieval context (if any), flags
- evaluation outputs: heuristic scores, hallucination flags, summary metrics

This is critical so comparisons remain consistent across pipelines.

### 4.4 Add config and settings handling
Create a central settings mechanism for:
- model provider selection
- default timeouts
- default pipeline choices
- domain location

---

## Phase 5 - RAG Indexing (Offline Build Step)

This phase ensures RAG is fast and reproducible.

### 5.1 Build a document loader for the domain
Implement a loader that:
- reads files from `documents_path`
- extracts raw text
- retains metadata (source file, section headers if possible)

Start with plain text/markdown/html first. Add PDFs later only if needed.

### 5.2 Implement chunking logic
Create chunking that:
- splits text into fixed-size chunks
- includes overlap
- attaches metadata to each chunk:
  - chunk id
  - source file
  - text

### 5.3 Choose embedding model (for indexing)
Select a single embedding model for MVP.

### 5.4 Write an indexing script (one-time build)
Create a script that:
- loads docs
- chunks docs
- embeds each chunk
- builds a FAISS index
- saves:
  - `index.faiss`
  - `chunks.json` (id -> text + metadata)
  - `index_meta.json` (embedding model id, chunk params, corpus hash, build timestamp)

### 5.5 Validate index quality early
Manually test retrieval:
- enter sample queries
- confirm retrieved chunks are relevant
If retrieval is bad, adjust:
- chunk size / overlap
- number of chunks returned (k)
- document selection

Do not move forward until retrieval is reasonable.

---

## Phase 6 - LLM Provider Layer (Generation)

### 6.1 Decide your generation model approach
For MVP:
- use a strong pre-trained LLM via API (recommended)

Make sure your system is provider-agnostic:
- the rest of the backend should call a single `generate()` interface.

### 6.2 Implement a provider adapter layer
Create a single interface that returns:
- generated text
- token usage (if available)
- any metadata you can use for cost estimation

### 6.3 Add rate limiting and timeouts (basic)
Avoid runaway usage by enforcing:
- request timeout per pipeline
- max tokens per response
- max concurrent requests (optional)

---

## Phase 7 - Implement Pipelines (Prompt-only + RAG)

### 7.1 Prompt-only pipeline
Define:
- a consistent system prompt template
- a user prompt format

Measure:
- latency

Collect:
- tokens + cost estimate

Return:
- standardized output object

### 7.2 RAG pipeline
Define:
- retrieval step:
  - query embedding
  - FAISS search
  - top-k chunk selection
- prompt assembly step:
  - include retrieved context
  - instruct model to use only provided docs

Measure:
- retrieval time (optional)
- generation time

Return:
- answer
- retrieved chunks (store top-k for UI display)
- latency + tokens + cost estimate

### 7.3 Concurrency design
Ensure both pipelines can run:
- in parallel for fairness
- under identical timeout constraints
- with the same base model + generation settings

---

## Phase 8 - Evaluation Engine (Comparison Logic)

This is the heart of the project.

### 8.1 Define evaluation dimensions
For MVP, evaluate:
- latency (ms)
- token usage
- cost estimate
- hallucination indicators
- basic quality heuristic

### 8.2 Implement domain-specific hallucination checks (dev docs)
Examples of dev-doc checks:
- detect mentions of APIs not present in docs
- detect parameter names not in retrieved context (for RAG)
- flag overconfident language without support

Start conservative:
- it's better to under-flag than over-claim.

### 8.3 Implement basic quality heuristics
Define a simple, explainable rubric, such as:
- does it contain required keywords?
- does it reference the correct component/function?
- does it provide a step-by-step solution?
- does it provide code-like structure when needed?

Do not attempt perfect grading. Be consistent and transparent.

### 8.4 Normalize and score outputs
Create consistent scoring across pipelines:
- same set of metrics returned for prompt-only and RAG
- unify into a summary object

### 8.5 Create an evaluation summary
Return an overall comparison summary:
- which approach performed best by metric
- what tradeoff was observed (latency vs correctness vs hallucination)

---

## Phase 9 - Run Orchestration and API Response

### 9.1 Implement the `run` orchestration flow
The `/run` request should:
- validate inputs
- load domain spec
- run chosen pipelines concurrently
- evaluate results
- return structured output

### 9.2 Add graceful failure behavior
If one pipeline fails:
- return the successful pipeline result
- include error information in metadata
Do not crash the whole request.

### 9.3 Add minimal logging
Log each run:
- timestamp
- domain
- query
- pipeline outputs metadata (including model + generation settings)
- evaluation flags/scores

Store in JSONL or SQLite for reproducibility.

---

## Phase 10 - Frontend UI (Comparison Interface)

### 10.1 Create the UI skeleton
UI should include:
- domain dropdown
- pipeline toggles
- query input
- run button
- results section

### 10.2 Results display requirements
Show:
- prompt-only answer
- RAG answer
- metrics per pipeline:
  - latency
  - tokens
  - cost estimate
  - flags (hallucination warnings)
- retrieved chunks viewer for RAG
  - expandable "show evidence" panel

### 10.3 Add UX essentials
- loading state
- error display if pipeline fails
- disable run button while running

---

## Phase 11 - Local End-to-End Testing

### 11.1 Create a test query set
Create 20-50 queries in a file:
- easy questions
- tricky edge cases
- ambiguous questions
- questions that tempt hallucination

Treat this as a regression suite and re-run it before each release.

### 11.2 Run a full evaluation pass locally
Verify:
- concurrency works
- results are consistent
- retrieval is relevant
- evaluation flags make sense

### 11.3 Tune the system
Iterate on:
- chunking parameters
- retrieval_k
- prompt templates
- evaluation heuristics

Stop tuning when you see stable, explainable tradeoffs.

---

## Phase 12 - Packaging for Deployment

### 12.1 Backend deployment readiness checklist
Ensure:
- environment variables are documented
- API keys are not committed
- index files are available for deployment (per Phase 0.5)
- health endpoint works

### 12.2 Confirm index build strategy (MVP: commit artifacts to repo)
Verify:
- `domains/<domain>/artifacts/` contains `index.faiss`, `chunks.json`, and `index_meta.json`
- artifacts are committed to the repo (MVP choice)
- the deployment includes these artifacts (e.g., via building the backend from the repo without excluding them)

### 12.3 Add Docker support for backend
Prepare backend to run in a container:
- install deps
- run server
- include index artifacts or build them

---

## Phase 13 - Backend Deployment

### 13.1 Choose a hosting platform
Pick one platform that supports:
- environment variables
- Python web services
- logging
Examples: Render, Fly.io, Railway

### 13.2 Configure environment variables
Set:
- LLM API key
- any model/provider settings
- domain and index paths (if needed)

### 13.3 Deploy backend and verify endpoints
Verify:
- `/health` returns OK
- `/domains` lists domains
- `/run` returns results for both pipelines

Confirm logs show run records.

---

## Phase 14 - Frontend Deployment

### 14.1 Choose a frontend host
Examples: Vercel, Netlify

### 14.2 Configure API base URL
Set frontend environment variable pointing to your deployed backend.

### 14.3 Deploy and verify end-to-end
Verify:
- UI loads
- domains populate
- queries run successfully
- outputs and metrics display correctly
- RAG evidence panel works

---

## Phase 15 - Documentation and Portfolio Polish

### 15.1 Update README to match shipped product
Include:
- what problem it solves
- architecture diagram
- how to run locally
- how to deploy
- how to add a new domain
- example comparison table with real results
- limitations and honest evaluation notes

### 15.2 Add an Evaluation doc
In `docs/EVALUATION.md`, document:
- metrics used
- heuristics used
- why they are imperfect
- failure modes observed
- improvements planned

This is a strong research-engineer signal.

### 15.3 Record a demo
Record a 60-120 second demo showing:
- prompt-only hallucination
- RAG grounded answer with evidence
- metrics/tradeoffs shown clearly

---

## Phase 16 - Optional Extensions (Only After MVP Is Deployed)

### 16.1 Add Fine-tuning pipeline
Only after everything above is stable:
- build a small dataset of domain Q&A
- fine-tune a small model
- evaluate against prompt-only and RAG
- quantify whether it is worth it

### 16.2 Add second domain (proof of swap-ability)
Example:
- React docs domain
Show in README:
- "Added new domain by creating data + config + evaluation rules only."

### 16.3 Add caching for speed/cost
Cache:
- embeddings
- repeated queries
- repeated retrieval results
This improves latency and cost.

---

## Final "Done" Checklist

You are done when:
- [ ] Prompt-only and RAG both run and return structured results
- [ ] Pipelines run concurrently under identical generation settings
- [ ] Domain can be swapped via config
- [ ] UI compares outputs side-by-side with metrics and RAG evidence
- [ ] Backend and frontend are deployed
- [ ] README includes architecture + results table + how to add a domain
- [ ] Demo video exists

If all boxes are checked, this is a legitimate flagship project.
