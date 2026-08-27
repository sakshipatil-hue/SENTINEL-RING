# Rules.md — Sentinel Ring

Hard constraints for this project. These are non-negotiable — any code, prompt, or design change must be checked against this list before merging. If a feature idea conflicts with a rule here, the rule wins.

## Track 2 Compliance Rules (from Razorpay Buildathon)

1. **Defense-only.** The system must never contain functionality that could itself be used to commit fraud, evade detection, or attack another system. No "offense-capable" code, ever — not even as a demo of "what attackers do."
2. **No autonomous irreversible actions.** The agent may only output one of: `AUTO-MONITOR`, `FLAG-FOR-REVIEW`, `MANUAL-REVIEW`. It must never auto-block, auto-freeze, auto-reject, or auto-charge. A human always makes the final irreversible call.
3. **Measured evaluation is mandatory.** Every claim of accuracy/precision/recall must come from an actual held-out test computed in code — never estimated, assumed, or asserted without a metric.
4. **One loss category, done well.** Stay scoped to coordinated abuse (return/promo/card-testing rings). Do not expand into unrelated fraud types just because the tech could generalize.

## Agent Design Rules

5. **The agent loop is capped at `MAX_AGENT_TOOL_CALLS` (6), enforced in code, not just prompted.** If the cap is hit before the agent reaches a decision, it must still output a structured result — defaulting to `MANUAL-REVIEW` with a reason noting the investigation was inconclusive within the call budget. Never let the loop silently fail or hang.
6. **`calculate_ring_risk()` is never an agent-callable tool.** It is computed deterministically upstream (in `src/network/`) and passed to the agent as fixed input context. The agent investigates and explains; it does not compute the core risk score itself.
7. **Agent tools are read-only.** None of `get_transaction()`, `get_customer_history()`, `get_connected_entities()`, `get_similar_cases()` may mutate any data. They fetch; they never write.
8. **Every agent output must be structured JSON matching the schema in Design.md — never free-form prose as the final answer.** Free text is fine as internal reasoning; the final output must validate against the schema before being returned by the API.
9. **Every agent decision must cite the evidence it used.** No recommendation without a populated `evidence` list — a bare risk tier with no evidence is a rule violation.

## Data & Evaluation Rules

10. **Ground truth (`rings.json`, `is_ring_member`, `ring_id`, `is_fraud_individual`) is never fed into the ML detector or the network detection features.** It exists solely for injection and post-hoc evaluation. If any model input accidentally includes a ground-truth column, that's a data leak — fix before proceeding.
11. **The ML detector and Network detector must be evaluated separately, each on its own held-out data, before being fused at the agent layer.** Don't report a single blended metric that hides which stage is actually doing the work.
12. **Real IEEE-CIS data retains its original `isFraud` label for lone/individual fraud.** Do not overwrite or reinterpret it. Injected ring labels are a separate, additional field — never conflated with the original column.
13. **Source data must be credited in the README** (IEEE-CIS Fraud Detection, Kaggle/Vesta) — required for using competition data in a public repo.

## Engineering Rules

14. **No stage may silently swallow an error.** If `generation`, `features`, `ml_detector`, `network`, or `agent` fails, it must raise/log clearly — never return a default/empty result that masks a broken pipeline.
15. **Every module (`src/generation`, `src/features`, `src/ml_detector`, `src/network`, `src/agent`, `src/api`) must be independently runnable/testable** — no module should require the full pipeline to run just to sanity-check its own output.
16. **Config values (volumes, seed, paths, `MAX_AGENT_TOOL_CALLS`) live only in `src/config.py`.** No hardcoded magic numbers duplicated elsewhere — if a constant needs to change, it should change in exactly one place.
17. **Commit at the end of each build day** (per Phases.md), even if incomplete — small, reviewable commits over one giant end-of-project commit.

## Scope Discipline Rules

18. **No new agent tools beyond the 4 defined**, unless a full day of buffer remains and the core pipeline (Days 1–11 in Phases.md) is complete and stable.
19. **No framework additions (LangChain, LangGraph, etc.) unless raw function-calling with the Groq API proves insufficient after a real attempt.** Simpler stack first.
20. **If behind schedule, cut polish before cutting correctness.** A smaller but honestly-evaluated pipeline beats a feature-complete but unverified one.
