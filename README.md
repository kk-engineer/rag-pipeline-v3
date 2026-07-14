# RAG Pipeline

A retrieval-augmented generation pipeline built with production-grade engineering discipline — deterministic synthesis, multi-layer hallucination defense, and full observability — running deliberately on lightweight, self-hostable infrastructure rather than managed cloud services.

This is a portfolio project. The goal is not "a RAG demo that answers questions" — dozens of those exist. The goal is a system where every reliability, safety, and cost decision is explicit, configurable, and measured, so it can serve as a concrete reference point in a conversation about building this kind of system for a real team.

> **NOLLM Agents philosophy:** LLMs are reserved for synthesis and generative reasoning only. Every classification, scoring, routing, and verification decision in this pipeline is handled by a small, deterministic, purpose-built model. See [Architecture](#architecture) for why this matters for both cost and reliability.

---

## Demo

- 🎥 [Video walkthrough](#) *(local run, M1 Mac — link added on completion)*
- 🤗 [HF Spaces live demo](#) *(lightweight profile — added in Phase 10)*
- 📊 [Evaluation report](data/golden_dataset/reports/) — Precision@k, Recall@k, Faithfulness, and more, computed against a hand-built golden dataset, not asserted

---
# Initial Setup
```shell
uv pip install -e .
uv sync
```

## Run command

```shell
PYTHONPATH=.:src uv run gradio interfaces_app/gradio_app/app.py
```

## Why this exists

Most public RAG repos stop at "chunk, embed, retrieve, generate." That's a prototype, not a system a team could put in front of real users or real documents. This project builds out the parts that actually determine whether a RAG system is trustworthy:

- **Determinism** — same input, same output, every time (`temperature=0`, fixed seed, structured schema-validated generation).
- **Defense in depth against hallucination** — a circuit breaker before generation (don't answer without relevant context), and a sentence-level NLI verification gate after generation (don't ship an answer the context doesn't support), with a capped, cost-bounded self-correction loop in between.
- **Multi-layer safety** — prompt-injection detection on every query before it reaches any model, with a configurable content-safety layer for teams that need it (see [Safety Design](#safety-design) for why this is a toggle, not always-on, in this build).
- **Real numbers, not claims** — every quality claim in this README is backed by a metrics report generated from an actual eval run against a golden dataset, checked into the repo.
- **Cost-aware model tiering** — every stage uses the smallest model that can do the job; the large generative model is called exactly once per query, at the end, for synthesis only.

---

## Architecture

```
Query
  │
  ▼
[1] Query Contextualization  ── statistical coreference resolution (LLM fallback only on low confidence)
  │
  ▼
[2] Routing & Injection Defense  ── PromptGuard → rule-engine → intent classifier → confidence gate
  │
  ▼
[3] Hybrid Retrieval  ── dense (Chroma) + sparse (BM25) → RRF fusion → cross-encoder rerank → circuit breaker
  │
  ▼
[4] Compression & Budgeting  ── sentence-level relevance filtering → token budget enforcement
  │
  ▼
[5] Deterministic Synthesis  ── NIM-hosted LLM, temperature=0, fixed seed, schema-validated JSON output
  │
  ▼
[6] Verification & Self-Correction  ── citation check → DeBERTa-MNLI sentence-level gate → capped retry
  │
  ▼
Answer + Citations
```

Every stage above is wired through an internal `Tracer` interface into Langfuse + OpenSmith, and every threshold is tunable in `config.toml` rather than hardcoded — see [Configuration](#configuration).

### Why the ingestion pipeline matters as much as the query pipeline

Documents are anonymized (Microsoft Presidio) and chunked using semantic-drift boundary detection (a small sentence-embedding model, not fixed-size splitting and not an LLM call) before anything is indexed. Every chunk carries immutable metadata (`tenant_id`, `doc_version_uuid`, `filename`, `page_number`) enabling multi-tenant isolation at the database query level, not just at the application layer.

### Design pattern usage (not decorative — structurally load-bearing)

| Pattern | Where | What it buys |
|---|---|---|
| **Strategy** | Retrieval method, synthesis provider, safety gates | Swap Chroma → BM25 → hybrid, or NIM → local MLX fallback, via config — zero code changes |
| **Adapter** | Every third-party tool (Chroma, Redis, Presidio, NIM, Langfuse) | Internal code never depends on a vendor SDK directly — a real, tested "swap this for pgvector later" claim |
| **Composite** | Safety gates | Add/remove a gate (e.g., toggle content-safety layer) by editing config, not code |
| **Chain of Responsibility** | Query intake (contextualize → route → gate → confidence-check) | Each stage can halt the chain — the natural shape for a sequence of gates |
| **Builder** | Pipeline assembly per profile (`vanilla` / `reliable` / `chaos`) | Three genuinely different pipeline graphs from one codebase |
| **Decorator** | Tracing | Any stage gets full observability without its own logic knowing tracing exists |
| **Observer** | Dual tracing sinks (Langfuse + OpenSmith) | Adding a third sink later is a registration, not a rewrite |
| **Factory** | Component instantiation from `config.toml` | The only place environment-branching logic is allowed to live |

Full interface/adapter boundary lives in `src/rag_pipeline/core/interfaces/` (the ports) and `src/rag_pipeline/adapters/` (the implementations) — this is also what makes the Gradio-now, FastAPI-later, HF-Spaces-later roadmap a real architectural claim rather than a to-do list.

---

## Safety Design

Two distinct concerns are handled by two distinct, purpose-built models — they are not redundant with each other, and neither is redundant with the base LLM's own RLHF safety training:

- **PromptGuard** (`Llama-Prompt-Guard-2-86M`) — detects prompt injection / jailbreak attempts on every query, always on. This is the higher-priority gate for a RAG system specifically, because it also runs against **retrieved document content**, not just the user's query — a PDF containing hidden injected instructions is a realistic RAG-specific attack surface that query-only scanning misses entirely.
- **LlamaGuard** (`Llama-Guard-3-1B`) — content-harm-category classification (violence, self-harm, hate, etc.). **Configurable, off by default in the local profile** for RAM headroom on a 16GB machine running multiple concurrent models; on by default in the cloud/Spaces profile. This is implemented, not just claimed — see the eval report for a run with it enabled.

Relying on the base LLM's own refusal behavior alone was considered and rejected: it produces no structured, loggable signal, and it degrades unpredictably under the `chaos` tier's smaller local fallback model, which is typically less safety-tuned than the primary NIM-hosted model.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Package management | `uv` | Fast, lockfile-based, single source of truth |
| Vector store | ChromaDB (embedded) | Zero external infra — a deliberate "clone and run" choice |
| Sparse retrieval | `rank_bm25` | Pure Python, no server |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Small, CPU-fast, reused across two pipeline stages |
| Synthesis | NVIDIA NIM (`llama-3.3-70b-instruct` / Nemotron) free tier, MLX-LM local fallback | Heaviest model runs in the cloud; local fallback is Apple-Silicon-native, not generic torch+MPS |
| PII | Microsoft Presidio + spaCy | Industry-standard, self-hostable |
| Injection/safety | PromptGuard, LlamaGuard (toggleable) | See Safety Design above |
| Verification | `deberta-v3-base-mnli` | Purpose-built entailment classifier — don't ask the generator to grade its own homework |
| Memory | Redis | Idempotency + sliding-window conversation history |
| Observability | Langfuse + OpenSmith (SQLite-based) | Dual sinks via Observer pattern, zero external infra required |
| Eval | PromptFoo, DeepEval, Garak | See [Evaluation](#evaluation) for exactly when/how each runs |
| UI | Gradio | Phase 1-9; FastAPI added Phase 10 without touching pipeline internals |

---

## Getting Started

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone <repo-url>
cd rag-pipeline
uv sync

# Copy and edit config
cp config/config.toml.example config/config.toml
# set your NVIDIA NIM API key, profile = "local"

# Ingest the sample documents
uv run python scripts/ingest_pdfs.py --input data/pdfs/

# Launch the app
uv run gradio interfaces_app/gradio_app/app.py
```

Requires Python 3.11+, ~4GB free RAM for the local model stack (see [Model Sizing](#local-model-sizing) below), and an NVIDIA NIM API key (free tier) for synthesis. No GPU required.

---

## Configuration

Everything that could plausibly differ between local dev, the HF Spaces demo, or a before/after fine-tuning comparison lives in `config/config.toml` — nothing is hardcoded in `src/`.

```toml
[profile]
active = "local"  # local | spaces

[profile.local]
llama_guard_enabled = false
verification_model = "microsoft/deberta-v3-base-mnli"
synthesis_provider = "nim"  # nim | mlx_local

[profile.spaces]
llama_guard_enabled = true
verification_model = "microsoft/deberta-v3-base-mnli"
synthesis_provider = "nim"

[pipeline]
tier = "reliable"  # vanilla | reliable | chaos

[thresholds]
# Tuned against the golden dataset — see data/golden_dataset/reports/
chunk_drift_cosine = 0.35
injection_confidence_floor = 0.80
intent_confidence_floor = 0.75
retrieval_circuit_breaker = 0.55
correction_loop_max_attempts = 3
```

---

## Local Model Sizing (M1 / 16GB reference)

Synthesis runs via a cloud API call (NIM), so local RAM only ever holds the small specialist stack — and models are lazy-loaded per pipeline stage, not held resident simultaneously.

| Model | Role | Approx. size |
|---|---|---|
| `all-MiniLM-L6-v2` | chunk boundary detection | ~90MB |
| `en_core_web_sm` + Presidio | PII detection | ~50MB |
| `Llama-Prompt-Guard-2-86M` | injection detection | ~350MB |
| `Llama-Guard-3-1B` | content safety (toggleable) | ~2GB |
| MiniLM intent classifier | routing | ~90MB |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | rerank + compression | ~90MB |
| `fastcoref` | query contextualization | ~500MB |
| `deberta-v3-base-mnli` | verification | ~350MB |

---

## Evaluation

All quality claims in this repo are backed by a golden dataset (`data/golden_dataset/`, ~60-100 hand-verified query/context/answer triples drawn from the 3 source PDFs, covering factual, multi-hop, out-of-scope, adversarial, and ambiguous cases) and a reproducible eval run:

```bash
make eval
```

This runs PromptFoo + DeepEval against the golden dataset and produces a metrics report (Precision@k, Recall@k, MRR, NDCG, Faithfulness, Answer Completeness, Factual Accuracy) checked into `data/golden_dataset/reports/`.

Garak red-teaming runs separately, against the live endpoint rather than the golden dataset (it probes for jailbreak surfaces, not answer quality):

```bash
make redteam
```

A before/after comparison — off-the-shelf models (Sprint 1) vs. fine-tuned intent classifier (Sprint 2) — is included, so improvement claims are demonstrated, not asserted.

---

## Project Structure

```
src/rag_pipeline/
├── prompts.py            # every prompt used anywhere in the project — single source of truth
├── config/                # config.toml loader
├── core/
│   ├── models.py          # Pydantic domain models
│   └── interfaces/        # Protocols — the ports (Chunker, Retriever, Synthesizer, ...)
├── adapters/               # concrete implementations — the adapters (Chroma, Presidio, NIM, ...)
└── factories/              # config-driven instantiation (Factory + Builder)
interfaces_app/
├── gradio_app/              # primary UI
└── fastapi_app/              # added Phase 10, calls the same core.pipeline
```

Full interface/adapter separation means a new frontend, a new vector store, or a new LLM provider is a new adapter file plus a config entry — not a pipeline rewrite.

---

## License

MIT
