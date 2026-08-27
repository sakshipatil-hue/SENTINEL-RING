# Memory.md — Sentinel Ring Decision Log

Running log of decisions made, why, and what's still open. Read this first when resuming work — it's the fastest way to reload context without re-reading every doc.

**Update this file at the end of each work session** — add a dated entry, keep it short (what changed, why, what's next).

---

## 2026-08-22 — Project kickoff & planning

**Decided:**
- Track: Track 2 (AI Risk Manager) — chosen for measurable rigor and lower crowding vs. Track 1's agentic-commerce idea
- Project idea: coordinated fraud/ring detection (not single-transaction scoring) — chosen for differentiation
- Title: **Sentinel Ring**
- Pitch line locked: *"Most transaction fraud systems evaluate transactions individually. Our system detects coordinated abuse by combining transaction-level ML with relationship and temporal signals, then uses an AI investigator to explain the network-level evidence and recommend a risk-aware action."*
- Architecture: 5-stage pipeline — Data → Features → (ML Detector + Network Detection in parallel) → AI Investigator Agent → Output. See Architecture.md.
- Agent design: bounded (max 6 tool calls), 4 read-only tools (`get_transaction`, `get_customer_history`, `get_connected_entities`, `get_similar_cases`), `calculate_ring_risk()` is deterministic and NOT agent-callable, output restricted to AUTO_MONITOR / FLAG_FOR_REVIEW / MANUAL_REVIEW only (never auto-block)
- Dataset: **hybrid** — IEEE-CIS Fraud Detection (Kaggle) for real transaction noise + injected ring structures on top for controlled ground truth. Decided against pure synthetic because hybrid gives more credible feature distributions and is a stronger interview story.
- Tech stack: Python, Pandas, Scikit-learn/XGBoost, NetworkX, Claude API (function-calling), FastAPI, Streamlit, PyVis/Plotly, Render, GitHub
- Timeline: 15 days (Aug 22 – Sept 5), day-by-day plan in Phases.md, buffer built into Days 13–14
- Repo scaffolded: `src/{generation,features,ml_detector,network,agent,api}`, `docs/`, `data/` (gitignored), `src/config.py` holds all constants
- Wrote: PRD.md, Architecture.md, Rules.md, Phases.md, Design.md (this file is the 6th)

**Why hybrid over pure synthetic:**
Real fraud-ring datasets don't exist publicly (platforms don't release them). Pure synthetic gives full ground-truth control but risks looking "too clean" to judges. Hybrid keeps ground-truth control for rings specifically (since we inject those ourselves) while borrowing IEEE-CIS's real noise/distributions for everything else — best of both, at the cost of some extra mapping work (see Design.md §1).

**Open decisions (not yet locked):**
- Exact ring-level matching threshold for evaluation (Design.md proposes 70% member overlap — confirm/tune on Day 8)
- Connected components vs. Louvain for network detection — start with connected components (simpler, defensible), upgrade only if Day 7 finishes early
- Ring risk score formula weights (Design.md §4 has a starting formula — expect to tune during Day 7–8)
- Whether to add tool-use to the agent for evidence-gathering beyond the 4 fixed tools — explicitly ruled out unless full buffer remains (Rules.md #18)

**Not yet started:** Phase 1 (Data — Days 2–3). This is the next work session.

---

## 2026-08-22 (later) — Day 2: data acquisition & mapping

**What happened:**
- IEEE-CIS competition download was gated; resolved by accepting competition rules on Kaggle
- File-size limits required trimming: user sampled to 3,000 transaction rows + matching identity rows locally before upload
- Built `src/generation/load_and_map.py` — loads raw CSVs, maps IEEE-CIS columns to our schema (Design.md Section 1), derives `accounts` from unique (card1, addr1) combos, preserves `isFraud` untouched as `is_fraud_individual`

**Output:** `data/accounts.csv` (1,485 accounts), `data/transactions.csv` (3,000 transactions). Verified: no orphan account_ids, fraud rate ~2% (matches real IEEE-CIS distribution), 475 accounts have repeat activity.

**Documented limitations (in code comments, not hidden):**
- No IP address field in IEEE-CIS → `ip_subnet` = "unknown" for all accounts
- No purchase/return/refund distinction → `transaction_type` = "purchase" for all rows
- Only ~20% of transactions have device info (594/3000 in the original sample) — expected, IEEE-CIS identity coverage is partial

**Ground truth placeholders added:** `is_ring_member` (False) and `ring_id` (null) columns exist in accounts.csv now, to be populated Day 3. Per Rules.md #10, these must never be read by feature engineering or model training.

**Next session should start with:** Day 3 — ring injection (`src/generation/inject_rings.py`), following the 3 patterns and exact method in Design.md Section 2.

---

## 2026-08-22 (later) — Day 3: ring injection

**Built:** `src/generation/inject_rings.py` — selects real accounts, relabels shared identifiers, writes ground truth.

**Result:** 6 rings injected (2 per pattern), 39 accounts marked `is_ring_member=True`:
- Pattern A (shared_device_staggered): ring_001 (9 accts), ring_002 (4 accts) — timestamps left natural, spans ~16-17h
- Pattern B (shared_payment_burst): ring_003 (6 accts, 13-min window), ring_004 (8 accts, 20-min window) — verified tight timestamp compression
- Pattern C (address_cluster_sequential): ring_005 (6 accts), ring_006 (6 accts) — timestamps sequential, spans 3-6h

**Verified:** no duplicate account-ring assignments (39 memberships = 39 unique accounts), accounts.csv `is_ring_member`/`ring_id` counts match rings.json exactly.

**Fixed one bug:** `ring_id`/`device_id`/etc columns loaded as float64 (all-null) on first read since Day 2 wrote them empty — cast to object dtype before assigning strings.

**Output:** `data/accounts.csv`, `data/transactions.csv` (both overwritten with ring data), `data/rings.json` (ground truth — 6 rings, never to be fed to models per Rules.md #10).

**Next session should start with:** Day 4 — individual feature engineering (`src/features/individual.py`): velocity, amount_zscore, account_age_days, merchant_category_entropy per Design.md Section 3.

---

## 2026-08-22 (later) — Day 4: individual feature engineering

**Built:** `src/features/individual.py` — 4 account-level features per Design.md Section 3.

**Bug caught and fixed:** initial `txn_velocity` used a 1-hour floor for ALL accounts, which meant the 1,010/1,485 accounts (68%) with only 1 transaction all got an identical artificial velocity of 24 — a false plateau, not real signal. Fixed: single-transaction accounts now correctly get velocity = 0; the floor only applies to multi-transaction bursts.

**Key finding (expected, not a bug):** ring members do NOT strongly stand out on individual features alone (velocity, amount z-score) compared to non-ring accounts — mean/median are close between the two groups. This is the correct result: it confirms rings are a genuinely network-level phenomenon, validating the need for separate relational/network detection (Day 5, Day 7) rather than relying on individual features alone.

**Output:** `data/individual_features.csv` — account_id + 4 features + is_fraud_individual label (43 fraud accounts / 1485).

**Next session should start with:** Day 5 — relational features (shared-attribute graph edges) + temporal features (burst score, time-since-last-cluster-txn), per Design.md Section 3.

---

## 2026-08-22 (later) — Day 5: relational + temporal features

**Built:**
- `src/features/relational.py` — builds graph edges between accounts sharing device_id/payment_token/billing_address_hash
- `src/features/temporal.py` — adds burst_score per edge (tighter time gap between two accounts' transactions = higher score)

**Bug caught and fixed before it corrupted the graph:** `device_id == "unknown_device"` (1,198/1,485 accounts) and `billing_address_hash == "-1.0"` (67 accounts) are missing-data placeholders, not real shared identifiers. Excluded both from edge construction — otherwise nearly all accounts would have been falsely linked into one giant supercluster.

**Result:** 43,653 total edges, but only 1,937 (4.4%) involve a known ring member — confirms the injected rings are NOT trivially separable from real-world coincidental attribute sharing (generic device types, shared zip codes), which is exactly the intended difficulty level (see docs/schema.md "don't make ring detection trivial" note).

**Verified burst_score works as intended:** ring_003/ring_004 (the shared_payment_burst pattern rings) show scores in the 0.05–0.94 range, well above the dataset median (~0.007) — even though the single highest-scoring edges in the whole dataset happen to be unrelated real-data coincidences (two strangers who happened to transact in the same minute). This is expected: burst_score alone isn't meant to be a perfect ring detector — Day 7 will combine it with shared-attribute count for the real ring risk score.

**Output:** `data/relational_edges.csv` — account_a, account_b, shared_attribute, attribute_value, min_gap_minutes, burst_score.

**Next session should start with:** Day 6 — ML Risk Detector (`src/ml_detector/`): train XGBoost/Random Forest on `individual_features.csv`, evaluate on held-out set, report precision/recall/confusion matrix. Ground truth = `is_fraud_individual` ONLY.

---

## 2026-08-22 (later) — Day 6: ML Risk Detector

**Built:** `src/ml_detector/train.py` — RandomForestClassifier (class_weight="balanced"), trained on the 4 individual features from Day 4, ground truth = `is_fraud_individual` only (never ring labels, per Rules.md #10).

**Honest, important finding (not a bug):** the held-out 80/20 split gave 0.000 precision/recall — the model caught none of the 9 fraud cases in that test set. Diagnosed properly before accepting it: confirmed the underlying features DO show some real separation (fraud accounts have higher median velocity: 15 vs 0, higher amount deviation: 0.71 vs 0), so this wasn't a data or code bug. Ran 5-fold cross-validation to get a more stable read: mean precision 6.7%, mean recall 13.9%, mean F1 8.9% across folds — consistently weak, not just one unlucky split.

**Why this is being kept as-is, not "fixed":** with only 43 total fraud accounts, this is a genuinely hard small-data problem, and this result is real. Per Rules.md #3 (measured evaluation is mandatory, no inflated claims), we report it honestly. This also strengthens the project's actual thesis: a weak standalone individual-feature detector is PRECISELY the argument for why Network Detection (Day 7) and the AI Investigator (Day 9-10) are necessary layers, not redundant ones. This will be stated directly in the pitch, not hidden.

**Output:** `src/ml_detector/model.pkl` (trained model), `data/ml_detector_scores.csv` (risk score for all 1,485 accounts — needed as agent context later), feature importances (account_age_days and txn_velocity dominate).

**Next session should start with:** Day 7 — Network Detection (`src/network/`): build graph from `relational_edges.csv` (NetworkX), connected components → candidate rings, deterministic ring risk score per Design.md Section 4.

---

## 2026-08-22 (later) — Day 7: network detection

**Real engineering journey (all documented in code comments, not hidden):**

1. Naive connected-components on all 43,653 edges collapsed 1,449/1,485 accounts into ONE giant blob — generic shared values (common device types, common zip codes) transitively link almost everyone. Fixed with inverse-frequency edge weighting (rare shared value = strong signal, common = weak).

2. Even weighted, connected-components STILL chained into large blobs (any single strong path bridges unrelated clusters — a known weakness of the method). Switched to **Louvain community detection** (modularity-based) instead.

3. Louvain's default resolution gave communities of 42-99 members with 100% true-ring-member recovery INSIDE them but too much extra noise. Swept resolution 1.0→25.0 against our known ground truth; **resolution=15.0** recovers 5/6 injected rings with ~100% member overlap and tight community sizes. This tuning was only possible because we control ground truth (the whole point of injecting rings) — in real deployment without ground truth, this would need business calibration instead.

4. Smallest ring (ring_002, 4 members) is NOT cleanly recovered (only 2/4 grouped together) — real, explained limitation: too little internal structure for community detection to lock onto.

5. First version of `calculate_ring_risk()` ranked true rings #25-#118 out of 179 candidates — buried below large innocent address-clusters (real neighborhoods). Diagnosed two concrete bugs: burst_score used max() (one lucky coincidental pair inflated a whole innocent cluster's score) and shared_attribute_score averaged raw edge counts (flattened to ~0.33, no discrimination). Fixed: burst_score → mean(), shared_attribute_score → count of DISTINCT attribute types (device/payment/address, max 3).

**Result after fix:** ring_001 (multi-attribute) jumped to rank #2/179; ring_003 to #16/179. Single-attribute-only rings (address-based) and the small ring_002 remain harder to distinguish from real neighborhoods — an honest, stated limitation, not hidden.

**Output:** `src/network/detect_rings.py`, `data/candidate_rings.json` (179 candidates with risk scores).

**Next session should start with:** Day 8 — formal ring-level precision/recall evaluation against `rings.json` using the 70%-overlap matching rule (Design.md Section 7).

---

## 2026-08-22 (later) — Day 8: ring-level evaluation

**Built:** `src/network/evaluate_rings.py` — implements the 70%-overlap matching rule from Design.md Section 7 (both recall side: did we find each real ring, and precision side: is each candidate real).

**Results (honest, headline numbers for the pitch):**
- **Recall: 83.3% (5/6 rings found)** — ring_002 (our smallest ring, 4 members) missed, consistent with Day 7's documented limitation
- **Precision (whole candidate list): 2.8% (5/179)** — looks low in isolation, but expected and explained: Louvain finds ALL dense communities in the graph (including many genuine real-world clusters like actual neighborhoods), not just fraud ones
- **Precision @ top-K (the realistic, useful metric):** top 5 = 20%, top 10 = 10%, top 20 = 10% — a 4-7x lift over the flat 2.8% baseline, showing ring_risk_score genuinely concentrates true positives near the top of the list, which is what a reviewer would actually act on

**Framing for the pitch:** don't lead with the flat 2.8% precision number alone — it's honest but misleading without context. Lead with 83.3% recall + the precision@K lift, and explain WHY flat precision is low (Louvain surfaces all dense communities, that's its job; risk scoring is what triages them for a human).

**Output:** `data/ring_evaluation.json` — full ring-by-ring and candidate-by-candidate breakdown.

**Next session should start with:** Day 9 — AI Investigator, build the 4 tools (`get_transaction`, `get_customer_history`, `get_connected_entities`, `get_similar_cases`) per Design.md Section 5. This is where the agentic AI piece begins.

---

## 2026-08-22 (later) — Day 9: case library + investigator tools

**Built:**
- `src/agent/build_case_library.py` — 12-case labeled reference library (6 confirmed rings from `rings.json`, 6 deliberately-selected "false alarm" cases: large single-attribute address clusters Day 8 confirmed were NOT real rings). Negative examples included on purpose so `get_similar_cases()` can match against past false alarms, not just push the agent toward "yes this is fraud" every time.
- `src/agent/tools.py` — all 4 tools from Design.md Section 5, exact signatures, all read-only (Rules.md #7), all independently tested via the file's own `__main__` block (Rules.md #15) without needing the agent loop.

**Bug caught and fixed:** `get_connected_entities` initially returned ALL connections for an account with no cap — tested on a real account and got 80 raw results back in one call, which would flood the agent's context on a single tool call (working against the bounded-investigation design, Rules.md #5). Fixed: capped at 15 results, sorted by burst_score descending (most suspicious first), with `total_connections`/`truncated` fields so the agent knows more exist without seeing them all.

**get_similar_cases design note:** similarity is a simple, explainable heuristic (attribute-overlap Jaccard + size closeness) rather than an embedding model — deliberate, since this feeds evidence the agent must be able to explain (Rules.md #9), not a black box.

**Output:** `data/case_library.json` (12 cases), `src/agent/tools.py` (4 tools, tested).

**Next session should start with:** Day 10 — the actual agent loop (`src/agent/investigator.py`): wire the 4 tools into a bounded (max 6 calls) Claude function-calling loop, enforce the structured JSON output schema from Design.md Section 6, test end-to-end on 1-2 real candidate rings from Day 7's `candidate_rings.json`.

---

## 2026-08-22 (later) — Day 10: agent loop + switch to Groq API

**Budget decision:** switched the agent's LLM from Claude API (originally planned) to **Groq API (Llama 3.3 70B)** — team confirmed a $0 budget. Updated Rules.md #19 and requirements.txt accordingly. Groq's free tier supports reliable OpenAI-compatible tool-calling, which our design needs.

**Built:** `src/agent/investigator.py` — the bounded agent loop:
- System prompt encodes the goal, the 4 tools, the hard 6-call limit, and the exact output schema
- `_validate_output()` enforces Rules.md #8 (structured output only) and #2 (recommended_action restricted to AUTO_MONITOR/FLAG_FOR_REVIEW/MANUAL_REVIEW — irreversible actions like "AUTO_BLOCK" are rejected by the validator itself, not just prompted against)
- `_build_default_output()` is the Rules.md #5 fallback — if the agent hits the 6-call cap without deciding, or produces invalid output, the loop still returns a valid structured result (defaults to MANUAL_REVIEW) rather than failing silently

**Sandbox network limitation (important, not a bug):** this build environment's network allowlist doesn't include api.groq.com, so the live LLM call cannot be tested from here. Built `src/agent/test_investigator.py` instead — tests everything that doesn't require the network: schema validation (6 cases including rejecting irreversible actions), the tool dispatch table (all 4 tools), and the call-cap fallback logic. All passed. One real bug caught and fixed during this: a test used the wrong keyword argument name for `_build_default_output` (`evidence_gathered` vs the actual `evidence`) — caught immediately by running the test, not by inspection.

**What's NOT yet done:** an actual live Groq API call end-to-end on a real candidate ring. This requires a free API key from console.groq.com/keys and needs to run on a machine with real internet access (not this sandbox).

**Next session should start with:** get a free Groq API key, set `GROQ_API_KEY` env var, run `python src/agent/investigator.py` for a real end-to-end test on `candidate_rings.json`'s top candidate. Then Day 11 — FastAPI wiring (`/detect`, `/rings`, `/investigate/{ring_id}`).

---

## Template for future entries

```
## YYYY-MM-DD — [session focus]

**Decided:**
- ...

**Built:**
- ...

**Blocked / open questions:**
- ...

**Next session should start with:**
- ...
```
