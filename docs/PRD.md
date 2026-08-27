# PRD.md — Sentinel Ring

## 1. Product Summary
Sentinel Ring is a coordinated-fraud detection system for the Razorpay AI Buildathon (Track 2: AI Risk Manager). It detects **rings of coordinated abuse** — not just individually suspicious transactions — by combining transaction-level ML, relational/temporal network signals, and a bounded AI investigator agent that produces explainable, auditable recommendations for human reviewers.

## 2. Problem
Most fraud systems score transactions in isolation. This misses coordinated abuse: multiple accounts sharing devices, payment instruments, or addresses, acting together to exploit returns, promos, or checkout flows. Each transaction looks clean alone; the pattern only appears when viewed as a network. Review teams are bottlenecked not by detection speed but by **verification capacity** — they need trustworthy evidence, not just a risk score, to act confidently.

## 3. Goals
- G1: Detect individual-transaction anomalies with measured precision/recall on a held-out set
- G2: Detect coordinated rings using relational + temporal signals, evaluated against known ground truth
- G3: Produce explainable, evidence-backed recommendations via a bounded AI investigator agent
- G4: Never take an irreversible action autonomously — recommend only
- G5: Ship a working, demoable, publicly-repo'd system by Sept 5, 2026

## 4. Non-Goals
- NG1: Not a real-time production system — batch/demo scale is sufficient
- NG2: Not detecting all fraud types — scoped to coordinated ring abuse (return/promo/card-testing patterns)
- NG3: Not building autonomous blocking/freezing — defense-only, recommend-only (Track 2 hard rule)
- NG4: Not building a general-purpose agent framework — 4 fixed tools, bounded loop, one job

## 5. Users / Personas
- **Primary:** A fraud/risk reviewer at a payments platform who currently reviews flagged accounts one at a time and wants network-level evidence, not just a score
- **Secondary (for the hackathon):** Razorpay judges assessing technical depth, explainability, and responsible-AI design

## 6. Success Metrics
| Metric | Target | Stage |
|---|---|---|
| ML detector precision/recall on held-out set | Reported honestly, no fixed target — must be measured and defensible | ML Risk Detector |
| Ring-level precision/recall vs. injected ground truth | Reported honestly | Network Detection |
| Agent tool-call bound respected | 100% of runs ≤ 6 calls | AI Investigator |
| Agent output structure validity | 100% of runs produce valid structured JSON (risk tier + evidence + action + reason) | AI Investigator |
| End-to-end demo runs without manual intervention | Yes/No | Full pipeline |

## 7. Functional Requirements
- FR1: System ingests transaction/account data (hybrid: IEEE-CIS + injected ring structures)
- FR2: System computes individual + relational + temporal features
- FR3: System trains and evaluates an ML classifier for individual-transaction risk
- FR4: System builds a graph from shared identifiers and detects candidate rings
- FR5: System computes a deterministic ring risk score (size, density, shared-attribute strength, timing synchrony)
- FR6: System runs a bounded AI agent that investigates a candidate ring using 4 defined tools and produces a structured recommendation
- FR7: System exposes results via an API (`/detect`, `/rings`, `/investigate/{ring_id}`)
- FR8: System provides a demo dashboard visualizing risk scores and ring graphs

## 8. Non-Functional Requirements
- NFR1: Every model decision must be explainable (feature importance for ML; evidence list for agent)
- NFR2: Every agent run must be bounded and logged (audit trail)
- NFR3: System must run end-to-end on a single laptop, no paid infra beyond the LLM API
- NFR4: Code must be organized so each pipeline stage (`generation`, `features`, `ml_detector`, `network`, `agent`, `api`) is independently testable

## 9. Out of Scope for v1 (possible stretch goals, not required)
- Louvain community detection (stretch; connected components is baseline)
- Multi-turn conversational review interface for human reviewers
- Real-time streaming ingestion

## 10. Deliverables (per Track 2 rules)
- Public GitHub repository
- 5-minute pitch video
- Architecture diagram (see Architecture.md)
- Measured metrics (precision/recall at both transaction and ring level)
