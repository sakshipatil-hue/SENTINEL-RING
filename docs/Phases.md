# Phases.md — Sentinel Ring Build Timeline

15 days available: Aug 22 – Sept 5, 2026. Updated for the hybrid (IEEE-CIS + injected rings) dataset decision.

## Phase 0 — Foundation (Day 1) ✅ Complete
- [x] Repo scaffolded (`src/generation`, `features`, `ml_detector`, `network`, `agent`, `api`)
- [x] `docs/schema.md` — data schema + ring injection design
- [x] `src/config.py` — central constants
- [x] `requirements.txt`, `.gitignore`, git initialized
- [x] PRD.md, Architecture.md, Rules.md, Phases.md, Design.md, Memory.md

## Phase 1 — Data (Days 2–3)
**Day 2 — Acquire & map real data**
- Download IEEE-CIS `train_transaction.csv` (+ `train_identity.csv`)
- Filter/sample to a working subset (~8,000–10,000 rows)
- Map columns to schema: `card1`→`payment_token`, `addr1`→`billing_address_hash`, `DeviceInfo`→`device_id`, `TransactionDT`→`timestamp`, `TransactionAmt`→`amount`
- Derive `accounts` table from unique card/address combinations
- Preserve original `isFraud` as `is_fraud_individual`
- **Deliverable:** `data/accounts.csv`, `data/transactions.csv` (no injected rings yet)

**Day 3 — Inject ring structures**
- Select real transaction clusters (not fabricated rows) and relabel shared attributes to construct 5–8 rings across the 3 patterns (shared-device-staggered, shared-payment-burst, address-cluster-sequential)
- Tighten timestamps within chosen clusters to create burst signatures where the pattern calls for it
- Write `data/rings.json` (ground truth: ring_id, members, shared_attributes, pattern, burst_window)
- **Deliverable:** Full hybrid dataset with ground truth, ready for feature engineering

## Phase 2 — Features (Days 4–5)
**Day 4 — Individual features**
- Velocity, txn amount pattern/deviation, account age, merchant category pattern
- **Deliverable:** Feature table for ML detector (`src/features/individual.py`)

**Day 5 — Relational + temporal features**
- Shared-attribute edge lists (device/address/token)
- Burst score, time-since-last-cluster-transaction
- **Deliverable:** Graph-ready feature output (`src/features/relational.py`, `src/features/temporal.py`)

## Phase 3 — Detection (Days 6–8)
**Day 6 — ML Risk Detector**
- Train XGBoost/Random Forest on individual features (ground truth = `is_fraud_individual` only — NOT ring membership)
- Evaluate on held-out set: precision, recall, confusion matrix
- **Deliverable:** Trained model + honest metrics report

**Day 7 — Network Detection**
- Build graph from relational features (NetworkX)
- Connected components (baseline) → candidate rings
- Deterministic ring risk score (size, density, shared-attribute strength, timing synchrony)
- **Deliverable:** Candidate ring list + scores

**Day 8 — Ring-level evaluation**
- Define and run ring-level precision/recall against `rings.json` ground truth
- Decide + document exact matching rule (e.g., how much member overlap counts as a "detected" ring)
- **Deliverable:** Ring-level metrics report; Louvain community detection only if time allows

## Phase 4 — Agent (Days 9–10)
**Day 9 — Build & test tools**
- Implement `get_transaction()`, `get_customer_history()`, `get_connected_entities()`, `get_similar_cases()`
- Build small labeled case library (10–15 past rings) for `get_similar_cases()` to search against
- Test each tool independently
- **Deliverable:** 4 working, independently-tested tools

**Day 10 — Agent loop**
- Wire tools into bounded (≤6 calls) Claude function-calling loop
- Enforce structured JSON output (risk_tier, evidence, recommended_action, reason)
- Test end-to-end on 1–2 candidate rings
- **Deliverable:** Working agent producing valid structured output

## Phase 5 — Integration (Days 11–12)
**Day 11 — API**
- FastAPI: `/detect`, `/rings`, `/investigate/{ring_id}`
- Wire ML detector + network detection + agent together
- **Deliverable:** Working API, testable via curl/Postman

**Day 12 — Dashboard**
- Streamlit app: risk score tables, ring graph visualization (PyVis/Plotly), agent case file viewer
- **Deliverable:** Working demo UI

## Phase 6 — Hardening & Submission (Days 13–15)
**Day 13 — End-to-end testing**
- Run full pipeline start to finish, fix bugs, test edge cases (empty ring, agent hitting call cap, no similar cases found)
- **Deliverable:** Stable, reproducible pipeline

**Day 14 — Docs & deploy**
- Finalize README (with IEEE-CIS credit per Rules.md #13), architecture diagram export, deploy to Render
- **Deliverable:** Public repo finalized

**Day 15 — Pitch**
- Record 5-minute pitch video
- Final submission
- **Deliverable:** Submitted

## Buffer Policy
Days 13–14 double as slack. If any earlier phase overruns, absorb the delay there — never cut Day 15 (pitch video) short, since that's a direct evaluation input.
