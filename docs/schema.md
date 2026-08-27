# Data Schema — Sentinel Ring

## Design principle
Every table has fields that support THREE downstream uses:
1. Individual ML features (velocity, history, account-level anomaly)
2. Relational/graph features (shared identifiers → edges)
3. Ground truth labels (so evaluation is honest, not hand-waved)

---

## `accounts`
| Field | Type | Purpose |
|---|---|---|
| account_id | str (PK) | unique identifier |
| created_at | datetime | account age feature |
| device_id | str | relational signal (shared device → ring edge) |
| ip_subnet | str | relational signal |
| payment_token | str | relational signal (shared payment instrument) |
| billing_address_hash | str | relational signal (shared address) |
| is_ring_member | bool | **ground truth**, hidden from model, used only for eval |
| ring_id | str, nullable | **ground truth**, which injected ring this belongs to |

## `transactions`
| Field | Type | Purpose |
|---|---|---|
| transaction_id | str (PK) | unique identifier |
| account_id | str (FK) | links to accounts |
| timestamp | datetime | temporal features (burst detection) |
| amount | float | individual ML feature |
| transaction_type | enum | purchase / return / refund |
| merchant_category | str | individual ML feature |
| device_id | str | redundant w/ account but allows device-hopping detection |
| is_fraud_individual | bool | **ground truth** — lone bad actor, NOT ring-related |

## `rings` (ground truth reference table — for injection + eval, not fed to models)
| Field | Type | Purpose |
|---|---|---|
| ring_id | str (PK) | unique ring identifier |
| account_ids | list[str] | members of this ring |
| shared_attributes | dict | which attributes were used to link them (device/address/token) |
| injected_pattern | str | e.g. "promo_abuse", "return_fraud", "card_testing" |
| burst_window_minutes | int | how tightly clustered in time |

---

## Ring injection patterns (pick 3, generate 5-8 rings total across them)

1. **Shared device, staggered accounts** — 5-10 accounts, same device_id, created within days of each other, each makes 1-2 "clean-looking" transactions, but as a batch shows return-abuse pattern
2. **Shared payment instrument, burst timing** — 4-8 accounts share a payment_token, all transact within a 10-20 min window (card testing / promo abuse signature)
3. **Address clustering, sequential activity** — accounts share billing_address_hash, activity is sequential (one after another, not simultaneous) — mimics a single person operating multiple accounts serially

## Non-ring "noisy" individual fraud (important — don't make ring detection trivial)
Also inject some `is_fraud_individual = True` accounts that are NOT part of any ring — otherwise the ML detector's job becomes "detect rings" by proxy, which defeats the point of having two separate stages. These should be lone anomalies (unusual amount, odd hour, new account + high value) with no shared attributes to anyone else.

## Volume targets (Day 2-3 build)
- ~2,000-3,000 accounts
- ~8,000-12,000 transactions
- 5-8 injected rings (mix of the 3 patterns above), ring sizes 4-10 accounts each
- ~3-5% lone individual fraud (non-ring), rest clean

This is small enough to keep training/inference fast on a laptop, large enough that ring detection isn't trivially easy (signal has to be found among noise).
