# Product Requirements Document (PRD)
## Domain-Swappable LLM Decision System

---

## 1. Overview

### Product Name
LLM Decision System (working name)

### One-Line Summary
A domain-agnostic system that empirically compares LLM architectures (prompt-only, RAG, fine-tuned) on real tasks, enabling data-driven decisions based on quality, hallucination risk, latency, and cost.

---

## 2. Problem Statement

Organizations adopting LLMs face a recurring, high-impact decision:

> Which LLM architecture should we deploy for this domain and task?

Common choices include:
- Prompt-only usage of a general LLM
- Retrieval-Augmented Generation (RAG)
- Fine-tuning a domain-specific model

These decisions are often made heuristically, leading to:
- hallucinations in production
- unnecessary cost
- avoidable latency
- brittle systems under domain shift

There is no lightweight, reusable framework that allows teams to **empirically compare these approaches under identical constraints**.

---

## 3. Goals

### Primary Goals
- Enable side-by-side comparison of multiple LLM pipelines on the same input
- Measure quality, hallucination indicators, latency, and cost
- Support multiple domains without rewriting core logic
- Be interactive, deployable, and reproducible

### Secondary Goals
- Serve as an educational artifact for ML system design
- Demonstrate real-world ML evaluation practices

---

## 4. Non-Goals

- Training large language models from scratch
- Consumer-scale production traffic
- Perfect automatic grading of outputs
- Medical or legal decision-making systems

---

## 5. Target Users

### Primary Users
- ML Engineers
- Software Engineers working with LLMs
- Research Engineers evaluating architecture tradeoffs

### Secondary Users
- Students learning applied ML systems
- Hiring reviewers evaluating ML reasoning ability

---

## 6. User Stories

- As a user, I want to submit a domain-specific query and see how different LLM approaches respond.
- As a user, I want to compare responses based on correctness, hallucination risk, latency, and cost.
- As a developer, I want to add a new domain without modifying core pipeline code.
- As an engineer, I want reproducible metrics to justify architecture decisions.

---

## 7. Functional Requirements

### 7.1 Supported Pipelines
- Prompt-only baseline
- Retrieval-Augmented Generation (RAG)
- Fine-tuned model (optional, scoped)

Each pipeline must:
- Accept identical inputs
- Return structured outputs
- Report latency and token usage

---

### 7.2 Domain Abstraction
Domains must be plug-ins defined by:
- document corpus
- chunking configuration
- retrieval configuration
- evaluation heuristics

Adding a domain must not require changes to:
- pipeline logic
- evaluation engine
- frontend

---

### 7.3 Evaluation
The system must compute:
- heuristic correctness score
- hallucination indicators
- latency statistics
- token usage and cost estimates

Evaluation must be:
- consistent across pipelines
- domain-aware but system-agnostic

---

### 7.4 Frontend
The UI must:
- display answers side-by-side
- visualize latency and cost
- flag hallucinations or unsupported claims
- allow pipeline and domain selection

UI polish is secondary to clarity and interpretability.

---

## 8. Non-Functional Requirements

### Performance
- Pipelines should execute concurrently
- Each pipeline should respect configurable timeouts

### Reliability
- Failure of one pipeline must not block others
- Partial results must still be returned

### Extensibility
- New pipelines can be added with minimal changes
- New evaluation metrics can be injected modularly

---

## 9. MVP Scope

### Must-Have
- Prompt-only vs RAG comparison
- One real domain (developer documentation)
- Latency and cost measurement
- Deployed backend and frontend

### Nice-to-Have
- Fine-tuned pipeline
- Second domain
- Robustness testing (prompt perturbations)

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Scope creep | Strict MVP checklist |
| Subjective evaluation | Conservative heuristics + transparency |
| API cost | Rate limits, caching |
| Over-engineering | Domain-agnostic core, simple plugins |

---

## 11. Success Criteria

The product is considered successful when:
- It is deployed and usable end-to-end
- Domains can be swapped via configuration
- Metrics are consistently computed and displayed
- Tradeoffs between pipelines are clearly observable

### MVP Acceptance Criteria (Definition of Done)
- [ ] `POST /run` runs prompt-only and RAG concurrently under identical timeouts.
- [ ] Results are comparable: each pipeline returns answer, latency, token usage, and cost estimate (when available).
- [ ] RAG includes retrievable evidence (chunks) that can be shown in the UI.
- [ ] RAG artifacts are built offline and committed to the repo for MVP (no index build on service startup).
- [ ] Failure isolation works: one pipeline failing does not block other pipeline results.
- [ ] UI shows side-by-side answers with latency/tokens/cost and an expandable evidence panel for RAG.
- [ ] A fixed regression query set (20-50) exists and is re-run before deployment.

---

## 12. Future Extensions

- Automated benchmark datasets
- Model routing policies
- Offline batch evaluation
- Visualization of failure clusters

---

## 13. Resume-Ready Description

Designed and deployed a domain-agnostic LLM evaluation system comparing prompt-only, RAG, and fine-tuned pipelines, measuring quality, hallucination rate, latency, and cost to guide architecture decisions under real-world constraints.
