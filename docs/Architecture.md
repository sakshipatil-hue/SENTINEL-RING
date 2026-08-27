# Architecture.md — Sentinel Ring

## System Overview

```
┌──────────────────────┐
│ Data Layer             │  IEEE-CIS (real) + injected ring structures
│ (hybrid dataset)       │  → accounts.csv, transactions.csv, rings.json
└──────────┬────────────┘
           │
┌──────────▼────────────┐
│ Feature Engineering     │
│ - Individual features   │  velocity, txn amount pattern, account age,
│ - Relational features   │  shared-attribute edges (device/addr/token)
│ - Temporal features      │  burst score, time-since-last-cluster-txn
└──────┬────────┬────────┘
       │        │
┌──────▼───┐ ┌──▼────────────────┐
│ ML Risk   │ │ Network Detection   │
│ Detector  │ │ (NetworkX)           │
│           │ │                      │
│ XGBoost / │ │ Graph construction → │
│ Random    │ │ connected components│
│ Forest    │ │ (baseline) / Louvain │
│           │ │ (stretch) →          │
│ Output:   │ │ candidate rings →    │
│ per-txn/  │ │ deterministic ring   │
│ account   │ │ risk score           │
│ risk      │ │                      │
│ score     │ │ Output: ring         │
│           │ │ candidates + score   │
└──────┬────┘ └─────────┬────────────┘
       │                │
       └───────┬────────┘
               │  (both feed into agent as context)
      ┌────────▼─────────────────┐
      │ AI Investigator Agent      │
      │                            │
      │ Goal: investigate a         │
      │ candidate ring               │
      │                              │
      │ Tools (max 6 calls total):   │
      │ - get_transaction()          │
      │ - get_customer_history()     │
      │ - get_connected_entities()   │
      │ - get_similar_cases()        │
      │                              │
      │ Loop:                        │
      │ Observe → Investigate →      │
      │ Gather evidence → Assess →   │
      │ Decide                       │
      │                              │
      │ calculate_ring_risk() is     │
      │ NOT a tool — it's precomputed│
      │ upstream (deterministic,     │
      │ not left to LLM judgment)    │
      └────────┬─────────────────────┘
               │
      ┌────────▼─────────────────┐
      │ Structured Output           │
      │                              │
      │ risk_tier: HIGH/MEDIUM/LOW   │
      │ evidence: [...]              │
      │ recommended_action:          │
      │   AUTO-MONITOR /              │
      │   FLAG-FOR-REVIEW /           │
      │   MANUAL-REVIEW               │
      │   (never auto-block/freeze)  │
      │ reason: "..."                 │
      └────────┬─────────────────────┘
               │
      ┌────────▼─────────────────┐
      │ API Layer (FastAPI)         │
      │ /detect  /rings              │
      │ /investigate/{ring_id}       │
      └────────┬─────────────────────┘
               │
      ┌────────▼─────────────────┐
      │ Demo Dashboard (Streamlit)   │
      │ - Risk score tables           │
      │ - Ring graph visualization    │
      │   (PyVis/Plotly)               │
      │ - Agent case file viewer       │
      └────────────────────────────────┘
```

## Module Boundaries (maps to `src/` structure)

| Module | Responsibility | Depends on |
|---|---|---|
| `src/generation/` | Load IEEE-CIS, map to schema, inject ring structures, output ground truth | `src/config.py` |
| `src/features/` | Compute individual, relational, temporal features | `src/generation/` output |
| `src/ml_detector/` | Train/evaluate classifier, output per-account risk score | `src/features/` individual features |
| `src/network/` | Build graph, detect candidate rings, compute deterministic ring risk score | `src/features/` relational + temporal features |
| `src/agent/` | Define tools, run bounded agent loop, produce structured output | `src/ml_detector/` + `src/network/` outputs |
| `src/api/` | Expose FastAPI endpoints wiring all stages together | All above |
| (Streamlit app, top-level) | Demo UI | `src/api/` |

## Data Flow Contracts (so modules stay decoupled)

- `generation` → `features`: a single `transactions` DataFrame + `accounts` DataFrame + `rings.json` (ground truth, held out from all models)
- `features` → `ml_detector`: a feature matrix with individual-level columns only
- `features` → `network`: a feature matrix with relational/temporal columns only, plus raw shared-identifier columns for edge construction
- `ml_detector` → `agent`: a dict `{account_id: risk_score}`
- `network` → `agent`: a list of candidate rings, each `{ring_id, member_accounts, ring_risk_score, shared_attributes, timing_summary}`
- `agent` → `api`: structured JSON per investigated ring (see Design.md for exact schema)

## Key Architectural Decisions

1. **Two independent detectors (ML + Network), fused only at the agent layer.** This keeps each stage separately evaluable (own precision/recall) and avoids one signal masking the other's weaknesses.
2. **`calculate_ring_risk()` is deterministic, not an agent tool.** Keeps core detection reliable and reproducible; the agent's job is investigation and explanation, not scoring.
3. **Agent loop is capped at 6 tool calls.** Prevents unbounded reasoning loops — a stated, deliberate safety/reliability design choice, not a limitation.
4. **Agent can only recommend, never execute.** Satisfies Track 2's defense-only, non-offense-capable constraint unambiguously.
5. **Hybrid dataset (IEEE-CIS + injected rings).** Real transaction/feature noise for credibility; controlled ring ground truth for honest evaluation. See Design.md for the mapping approach.
