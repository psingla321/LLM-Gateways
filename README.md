# LLM-Gateways

A production-style **LLM Gateway** built with **LiteLLM** and **FastAPI**, demonstrating multi-provider routing, automatic fallbacks, response caching, cost tracking, PII guardrails, and per-team budgeting — all wrapped in a real-world use case: **insurance claim summarization**.

The gateway sits between your application and any LLM provider (Azure OpenAI, OpenAI, Groq, Anthropic, …) and applies the same set of policies to every request — so you get one consistent, observable, cost-controlled path to GenAI.

---

## Why a Gateway?

Calling LLMs directly from application code creates the same problems on every team:

- One provider goes down → your app goes down.
- Every developer rolls their own retry / cost logic.
- PII leaks into prompts because no one centralized the guardrail.
- Finance can't tell which team is burning the budget.
- Identical prompts pay full price every time.

This gateway centralizes all of that into a single pipeline.

---

## Features

| Capability               | What it does                                                                  |
| ------------------------ | ----------------------------------------------------------------------------- |
| **PII Masking**          | Regex-based detection of SSN / Email / Phone / DOB / Credit Card / IP / Aadhaar / PAN — redacted **before** the prompt leaves the gateway. |
| **Task Classification**  | Rule-based scoring (token count + complex/simple signal keywords) tags each request as `simple` or `complex`. |
| **Smart Routing**        | `simple` → cheap model (e.g. `gpt-4o-mini`).  `complex` → premium model (e.g. `gpt-4o`). |
| **Automatic Fallback**   | If the primary model fails / times out, the next model in the chain is tried automatically. |
| **Two-Layer Cache**      | In-memory (L1) + SQLite (L2), keyed by SHA-256 of the *masked* prompt — identical claims are served at $0 cost. |
| **Cost Tracking**        | Per-model pricing table; every call's USD cost is computed and logged.        |
| **Usage Logging**        | Every request → one SQLite row with team, user, tokens, latency, model, fallback flag, cache flag, cost. |
| **Explainability**       | Each response includes a structured `explainability` block: why this model, why this classification, what PII was masked, cache decision, cost comparison vs alternatives. |
| **Mock Mode**            | Runs end-to-end with realistic latency + token counts and no API keys — useful for demos and CI. |

---

## Repository Layout

```
LLM-Gateways/
├── api.py                       FastAPI app — exposes /api/claim, /api/stats, /api/teams, …
├── demo.py                      Rich-console walkthrough of 5 scenarios (simple, complex+PII, cache, fallback, multi-team load)
├── dashboard.py                 Streamlit dashboard — team budgets, model mix, cost trends
├── requirements.txt             Python deps (FastAPI, Streamlit, Plotly, Pandas, python-dotenv, Rich; LiteLLM optional)
├── usage.db                     SQLite — usage_log + response_cache tables (auto-created)
├── app-flow-human-made.svg      Architecture / flow diagram (SVG)
│
├── gateway/                     The gateway library
│   ├── __init__.py              Exports LLMGateway
│   ├── core.py                  Orchestrator: full 8-step pipeline + routing table + explainability builder
│   ├── classifier.py            Rule-based simple/complex task classifier with confidence scoring
│   ├── pii_masker.py            Regex PII detection + redaction (SSN, Email, Phone, DOB, CC, IP, Aadhaar, PAN)
│   ├── cache.py                 Two-layer (memory + SQLite) response cache, SHA-256 keyed
│   └── logger.py                SQLite UsageLogger — per-request rows + aggregation helpers
│
└── frontend/
    └── index.html               Single-page UI served by FastAPI at "/"
```

---

## Request Flow

Each call to `/api/claim` runs through this 8-step pipeline in [gateway/core.py](gateway/core.py):

```
                ┌──────────────────────────────────────────────────────────┐
                │                  Client (UI / API caller)                │
                └────────────────────────┬─────────────────────────────────┘
                                         │ POST /api/claim
                                         ▼
                ┌──────────────────────────────────────────────────────────┐
                │                FastAPI app (api.py)                      │
                │            ── singleton LLMGateway ──                    │
                └────────────────────────┬─────────────────────────────────┘
                                         │
                                         ▼
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                          LLM Gateway pipeline                               │
   │                                                                             │
   │   1. PII MASKING            mask_pii()  →  redact SSN/email/phone/...       │
   │                                                                             │
   │   2. CLASSIFICATION         classify_task()  →  simple | complex            │
   │                              (tokens + keyword signals + confidence)        │
   │                                                                             │
   │   3. CACHE LOOKUP           ResponseCache.get(masked_text)                  │
   │                                                                             │
   │           ┌──── HIT ───►   return cached response  ($0, ~0 ms)              │
   │           │                                                                 │
   │           └─ MISS ─► 4. ROUTING                                             │
   │                        ROUTING_TABLE[task_type] →  primary + fallback chain │
   │                                                                             │
   │                     5. LLM CALL  (LiteLLM)                                  │
   │                        ┌────────────────────────────────────────────────┐   │
   │                        │  try primary                                   │   │
   │                        │     │  fail / timeout                          │   │
   │                        │     ▼                                          │   │
   │                        │  try fallback #1                               │   │
   │                        │     │  fail                                    │   │
   │                        │     ▼                                          │   │
   │                        │  try fallback #2  → …  → raise if all exhaust  │   │
   │                        └────────────────────────────────────────────────┘   │
   │                                                                             │
   │                     6. COST CALC          _cost(model, tokens_in, out)      │
   │                     7. CACHE WRITE        ResponseCache.set(...)            │
   │                     8. USAGE LOG          UsageLogger.log(RequestLog(...))  │
   │                                                                             │
   └────────────────────────────┬────────────────────────────────────────────────┘
                                ▼
                ┌──────────────────────────────────────────────────────────┐
                │   Response: summary + model_used + cost + latency +      │
                │   cache_hit + fallback_used + explainability { … }       │
                └──────────────────────────────────────────────────────────┘

                          ┌──────────────────────────────┐
                          │       SQLite (usage.db)      │
                          │   • response_cache  (L2)     │
                          │   • usage_log       (audit)  │
                          └──────────────┬───────────────┘
                                         │
                                         ▼
                          ┌──────────────────────────────┐
                          │   dashboard.py (Streamlit)   │
                          │   team budgets, model mix,   │
                          │   cost trends, fallback rate │
                          └──────────────────────────────┘
```

A hand-drawn SVG version of this diagram lives at [app-flow-human-made.svg](app-flow-human-made.svg).

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. (Optional) configure live mode — otherwise gateway auto-detects mock mode
cp .env.example .env   # then add AZURE_API_KEY / AZURE_API_BASE / OPENAI_API_KEY / GROQ_API_KEY

# 3. Run the FastAPI gateway + frontend
uvicorn api:app --reload --port 8000
# open http://localhost:8000

# 4. Or run the console demo
python demo.py

# 5. Or open the analytics dashboard
streamlit run dashboard.py
```

### Mock vs. Live mode

`LLMGateway(mock_mode=...)` resolves to:

- `True`  → built-in mock responses with realistic latency + token counts (no API keys needed).
- `False` → real LLM calls via LiteLLM.
- `None` (default) → auto-detect: mock if no `AZURE_API_KEY` / `OPENAI_API_KEY` / `GROQ_API_KEY` / `ANTHROPIC_API_KEY` is set.

Override via the `MOCK_MODE` env var (`true` / `false`).

---

## API Endpoints

| Method   | Path             | Purpose                                                |
| -------- | ---------------- | ------------------------------------------------------ |
| `POST`   | `/api/claim`     | Process a claim through the full pipeline              |
| `GET`    | `/api/stats`     | Aggregate stats (totals, cache hit-rate, fallback-rate)|
| `GET`    | `/api/teams`     | Per-team spend / budget snapshot                       |
| `GET`    | `/api/models`    | Per-model usage breakdown                              |
| `GET`    | `/api/recent`    | Last 25 requests (for the live UI feed)                |
| `DELETE` | `/api/recent`    | Clear recent-request log                               |
| `GET`    | `/api/mode`      | Reports whether the gateway is in mock or live mode    |
| `GET`    | `/`              | Serves the single-page frontend                        |

Example:

```bash
curl -X POST http://localhost:8000/api/claim \
  -H "Content-Type: application/json" \
  -d '{
        "claim_text": "Minor fender bender in parking lot, no injuries. Bumper scratched.",
        "team": "claims-auto",
        "user_id": "demo_user"
      }'
```

---

## Configuration

Environment variables (all optional — defaults work in mock mode):

| Variable                      | Purpose                                         |
| ----------------------------- | ----------------------------------------------- |
| `MOCK_MODE`                   | `true` / `false` — force mode (else auto)       |
| `AZURE_API_KEY`               | Azure OpenAI key                                |
| `AZURE_API_BASE`              | Azure endpoint URL                              |
| `AZURE_API_VERSION`           | Defaults to `2024-02-01`                        |
| `AZURE_DEPLOYMENT_SIMPLE`     | Cheap-model deployment name (default `gpt-4o-mini`) |
| `AZURE_DEPLOYMENT_COMPLEX`    | Premium-model deployment name (default `gpt-4o`)    |
| `OPENAI_API_KEY`              | OpenAI key (LiteLLM fallback path)              |
| `GROQ_API_KEY`                | Groq key (used as final fallback)               |
| `ANTHROPIC_API_KEY`           | Optional Anthropic key                          |

---

## Tech Stack

- **FastAPI** + **Uvicorn** — gateway API
- **LiteLLM** — provider-agnostic LLM client (Azure / OpenAI / Groq / Anthropic / Gemini)
- **SQLite** — usage log + response cache (zero-config persistence)
- **Streamlit** + **Plotly** + **Pandas** — analytics dashboard
- **Rich** — pretty console output for the demo script
