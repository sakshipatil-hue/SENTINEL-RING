# ◆ Sentinel Ring

**🔴 Live demo: [sentinel-ring.onrender.com](https://sentinel-ring.onrender.com)**
*(Free-tier hosting — the app may take 30-60 seconds to wake up on first visit after a period of inactivity.)*


**Coordinated fraud-ring detection for the Razorpay AI Buildathon 2026 — Track 2: AI Risk Manager**

> Most transaction fraud systems evaluate transactions individually. Sentinel Ring detects *coordinated* abuse by combining transaction-level ML with relationship and temporal signals, then uses a bounded AI investigator agent to explain the network-level evidence and recommend a risk-aware action — reducing false positives and giving reviewers auditable evidence, not just a score.

## Screenshots

![Dashboard overview]SENTINEL-RING/Dashboard.png

---

## The problem

Fraud rings — multiple accounts sharing a device, payment instrument, or address to exploit returns, promos, or checkout flows — look clean one transaction at a time. The pattern only appears when you look at the *network*. Review teams aren't bottlenecked by detection speed; they're bottlenecked by **verification capacity** — they need trustworthy evidence, not just a risk score, to act with confidence.

## The solution — a 5-stage pipeline

```
Real transaction data (IEEE-CIS) + injected ring ground truth
        ↓
Feature Engineering (individual + relational + temporal)
        ↓
ML Risk Detector (XGBoost/RF)   +   Network Detection (Louvain community detection)
        ↓                                    ↓
              AI Investigator Agent (bounded, 4 read-only tools)
                        ↓
     Structured recommendation: risk tier, evidence, action — never an
                    irreversible action, always human-reviewable
```

Full architecture, design rationale, and every engineering decision (including the real bugs found and fixed along the way) are documented in [`docs/`](docs/).

## Results — reported honestly

- **83.3% recall** (5/6 injected rings correctly identified)
- **Precision looks low in isolation (2.8%)** because Louvain community detection surfaces *every* dense cluster in the graph — including genuine real-world groups (real neighborhoods sharing an address), not just fraud ones. This is by design: detection casts a wide net, and risk-scoring + the AI Investigator triage it down to what's worth a reviewer's time.
- **Precision @ top-20 by risk score is 4-7x higher than the flat baseline** — proof the risk-scoring is doing real prioritization work, not just listing everything.
- The individual-transaction ML detector alone caught almost nothing (see `docs/Memory.md`, Day 6) — which is precisely the argument for why network detection and the AI investigator exist as separate layers, not redundant ones.

We did not cherry-pick these numbers. Every metric, including the unflattering ones, is computed in code (`src/network/evaluate_rings.py`) and reported as-is — see `docs/Memory.md` for the full, dated build log, including three real bugs found and fixed during development.

## Why this is genuinely agentic, not just "an LLM call"

The AI Investigator (`src/agent/investigator.py`) runs a real **observe → investigate → gather evidence → assess → decide** loop:
- 4 read-only tools (`get_transaction`, `get_customer_history`, `get_connected_entities`, `get_similar_cases`) the agent chooses when and whether to call
- Hard-capped at 6 tool calls — a deliberate, stated reliability/safety bound, not a limitation
- `calculate_ring_risk()` (the core risk score) is **never** agent-callable — it's computed deterministically upstream, so the agent's job is investigation and explanation, never scoring itself
- Output is restricted to exactly three actions: `AUTO_MONITOR`, `FLAG_FOR_REVIEW`, `MANUAL_REVIEW` — the agent can never recommend an irreversible action, enforced by code-level validation, not just prompting
- Built with raw Groq function-calling (OpenAI-compatible tool-use), deliberately with no LangChain/LangGraph — every line of the agent loop is inspectable and defensible, not hidden inside a framework abstraction

## Tech stack

Python · Pandas · Scikit-learn · NetworkX + python-louvain · Groq API (Llama/gpt-oss, function-calling) · FastAPI · Streamlit · PyVis

## Dataset

Hybrid: real transaction data from [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Kaggle/Vesta) for realistic transaction-level noise and distributions, combined with deliberately injected coordinated-ring structures (3 patterns: shared-device, shared-payment-burst, shared-address-sequential) for controlled, honest ground truth — since no public dataset contains real labeled fraud rings. Full methodology in `docs/Design.md`.

## Project structure

```
sentinel-ring/
├── app.py                   # Streamlit dashboard (run this to see it live)
├── src/
│   ├── generation/           # Data loading, mapping, ring injection
│   ├── features/              # Individual, relational, temporal feature engineering
│   ├── ml_detector/            # ML risk detector (training + evaluation)
│   ├── network/                 # Graph-based ring detection + evaluation
│   ├── agent/                    # AI Investigator: tools, case library, agent loop
│   └── api/                       # FastAPI backend
├── docs/                     # PRD, Architecture, Rules, Phases, Design, Memory
│                              # (Memory.md is the full dated build log — start there)
└── data/                     # Generated data (not committed — see below)
```

## Running it locally

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Regenerate the full pipeline (or use the data/ files already included)
python src/generation/load_and_map.py
python src/generation/inject_rings.py
python src/features/individual.py
python src/features/relational.py
python src/features/temporal.py
python src/ml_detector/train.py
python src/network/detect_rings.py
python src/network/evaluate_rings.py
python src/agent/build_case_library.py

# Launch the dashboard
streamlit run app.py
```

**To run live AI investigations:** get a free API key at [console.groq.com/keys](https://console.groq.com/keys). Either set it as an environment variable (`GROQ_API_KEY`) before launching, or paste it directly into the dashboard's sidebar — it's used only for your session, never stored.

## Safety & compliance (Track 2 requirements)

- **Defense-only** — nothing in this system can itself be used to commit fraud or attack another system
- **No autonomous irreversible actions** — the agent only ever recommends; a human makes every final call
- **Measured, not asserted** — every accuracy claim is computed from code, including the ones that don't flatter the system
- **One loss category** — scoped deliberately to coordinated ring abuse, not a general-purpose fraud framework

Full rule set: [`docs/Rules.md`](docs/Rules.md)

## Team

Sakshi — B.Tech AI & ML, IIST Indore
