# Design.md — Sentinel Ring

Detailed design decisions and exact interface contracts. Where PRD.md says *what* and Architecture.md says *how the pieces connect*, this document says *exactly what each piece looks like*.

## 1. Dataset Mapping (IEEE-CIS → Sentinel Ring schema)

| IEEE-CIS column | Our field | Notes |
|---|---|---|
| `TransactionID` | `transaction_id` | direct |
| `TransactionDT` | `timestamp` | seconds-from-reference; convert to datetime with an arbitrary anchor date |
| `TransactionAmt` | `amount` | direct |
| `card1` | `payment_token` | treat as opaque identifier |
| `addr1` | `billing_address_hash` | treat as opaque identifier |
| `DeviceInfo` (from identity table) | `device_id` | join on `TransactionID`; many rows will be null — handle missing gracefully, don't drop |
| `ProductCD` / `card4` | `merchant_category` (approx) | best-effort mapping, document the approximation in code comments |
| `isFraud` | `is_fraud_individual` | preserved as-is, never modified |

**Account derivation rule:** an `account_id` = a unique `(card1, addr1)` combination. Document this rule directly in `src/generation/` code, since it's a modeling choice, not a given fact.

## 2. Ring Injection — Exact Method

For each of the 3 patterns, the process is: **select real rows, then relabel shared-identifier columns to create an artificial link.** Never fabricate a transaction row from nothing — this is what preserves the "hybrid" benefit.

- **Pattern A — shared_device_staggered:** pick 5–10 existing accounts with distinct real transactions; overwrite their `device_id` with one shared synthetic value; leave timestamps as-is (staggered, not bursty) — mimics one operator running multiple accounts on one device over time
- **Pattern B — shared_payment_burst:** pick 4–8 accounts; overwrite `payment_token` with one shared value; compress their `timestamp` values into a random 10–20 minute window — mimics card-testing/promo-abuse bursts
- **Pattern C — address_cluster_sequential:** pick 4–8 accounts; overwrite `billing_address_hash` with one shared value; order timestamps sequentially with small gaps (minutes to hours apart, not simultaneous) — mimics one person operating several accounts serially

After injection, write `rings.json`:
```json
{
  "ring_id": "ring_001",
  "pattern": "shared_payment_burst",
  "account_ids": ["acc_1044", "acc_2291", "..."],
  "shared_attributes": {"payment_token": "synthetic_token_A"},
  "burst_window_minutes": 14
}
```

## 3. Feature Definitions

**Individual (per account/transaction):**
- `txn_velocity` — transactions per account in trailing 24h window
- `amount_zscore` — deviation from that account's own historical mean amount
- `account_age_days` — days since `created_at` (derived: first transaction timestamp per account)
- `merchant_category_entropy` — diversity of merchant categories used (low diversity + high velocity is a signal)

**Relational (graph edges):**
- Edge exists between account A and B if they share `device_id`, `payment_token`, OR `billing_address_hash`
- Edge weight = number of shared attributes (1–3)

**Temporal:**
- `burst_score` — for a candidate cluster, `1 / (time span of cluster's transactions in minutes + 1)` — higher = tighter burst
- `time_since_last_cluster_txn` — gap between an account's transaction and the nearest transaction from another account in its cluster

## 4. Ring Risk Score (Deterministic — `calculate_ring_risk()`)

Computed in `src/network/`, NOT by the agent. Simple weighted formula, each component normalized 0–1:

```
ring_risk_score = (
    0.3 * size_score +              # larger rings = higher risk, capped
    0.3 * density_score +            # edge density within the cluster
    0.25 * shared_attribute_score +  # avg edge weight (1-3 shared attrs)
    0.15 * burst_score                # from temporal features
)
```

Weights are a starting point — tune during Day 7–8 based on what separates real injected rings from coincidental overlaps in the eval set. Document any weight changes with a one-line reason in code comments.

## 5. Agent — Exact Tool Signatures

```python
def get_transaction(transaction_id: str) -> dict:
    """Returns full transaction record."""

def get_customer_history(account_id: str) -> dict:
    """Returns an account's transaction history summary + individual risk score."""

def get_connected_entities(account_id: str) -> dict:
    """Returns accounts sharing device/payment/address with this account, and which attribute(s)."""

def get_similar_cases(ring_pattern_summary: dict) -> list[dict]:
    """Returns top-k similar past labeled cases from the case library, with their outcomes."""
```

All four are **read-only** (Rules.md #7). None accept a write/mutate parameter.

## 6. Agent Output Schema (Design.md is the source of truth for this — Rules.md #8 enforces it)

```json
{
  "ring_id": "ring_001",
  "risk_tier": "HIGH",
  "evidence": [
    "8 accounts sharing 1 payment instrument",
    "3 shared devices across the cluster",
    "12 transactions within a 14-minute window"
  ],
  "recommended_action": "MANUAL_REVIEW",
  "reason": "High network-level risk score, but no individual account shows historical fraud confirmation — insufficient evidence for automatic escalation to FLAG_FOR_REVIEW.",
  "tool_calls_used": 5,
  "hit_call_cap": false
}
```

`recommended_action` enum is strictly: `AUTO_MONITOR`, `FLAG_FOR_REVIEW`, `MANUAL_REVIEW` — nothing else is valid (Rules.md #2).

## 7. Evaluation Design

**ML Detector (Day 6):** standard train/held-out split (e.g., 80/20, stratified on `is_fraud_individual`). Report precision, recall, F1, confusion matrix. Ground truth = original IEEE-CIS `isFraud` only.

**Network Detection (Day 8):** ring-level evaluation against `rings.json`. Matching rule (decide exact threshold during Day 8, document it here once fixed):
- A detected candidate ring counts as a **true positive** if it shares ≥ 70% of member accounts with a real injected ring
- A real injected ring with no matching candidate counts as a **false negative**
- A candidate ring with no real-ring match counts as a **false positive**
- Report ring-level precision/recall using this rule

## 8. Failure Modes to Handle Explicitly

- Agent hits the 6-call cap before deciding → still return structured output, `recommended_action: MANUAL_REVIEW`, `hit_call_cap: true`, reason states investigation was inconclusive within budget
- `get_similar_cases()` finds nothing similar → return empty list, agent must still produce a valid decision, not error out
- A candidate ring has only 2 members (borderline) → still scored and passed through, not silently dropped — let the risk score and agent judgment handle borderline cases, don't hardcode a minimum size filter that hides them
