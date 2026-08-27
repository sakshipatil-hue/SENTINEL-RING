"""
src/agent/tools.py

Day 9 (part 2): The 4 AI Investigator tools (Design.md Section 5).
ALL FOUR ARE READ-ONLY -- none accept a write/mutate parameter (Rules.md #7).
These are the exact tool signatures the agent (Day 10) will call.

Each function is directly testable in isolation (Rules.md #15) -- see the
__main__ block below for standalone tests of all 4 before wiring them into
an actual agent loop.
"""

import os
import sys
import json

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import DATA_DIR

# Loaded once at module level -- tools are read-only and stateless, so
# there's no reason to re-read CSVs on every call within a single run
_transactions = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"), parse_dates=["timestamp"])
_accounts = pd.read_csv(os.path.join(DATA_DIR, "accounts.csv"))
_edges = pd.read_csv(os.path.join(DATA_DIR, "relational_edges.csv"))
_ml_scores = pd.read_csv(os.path.join(DATA_DIR, "ml_detector_scores.csv"))
_case_library = json.load(open(os.path.join(DATA_DIR, "case_library.json")))


def get_transaction(transaction_id: str) -> dict:
    """
    Returns full transaction record. Read-only.
    Design.md Section 5: `def get_transaction(transaction_id: str) -> dict`
    """
    row = _transactions[_transactions["transaction_id"].astype(str) == str(transaction_id)]
    if row.empty:
        return {"error": f"transaction_id {transaction_id} not found"}
    r = row.iloc[0]
    return {
        "transaction_id": str(r["transaction_id"]),
        "account_id": r["account_id"],
        "timestamp": str(r["timestamp"]),
        "amount": float(r["amount"]),
        "transaction_type": r["transaction_type"],
        "merchant_category": r["merchant_category"],
        "device_id": r["device_id"],
        "is_fraud_individual": bool(r["is_fraud_individual"]),
    }


def get_customer_history(account_id: str) -> dict:
    """
    Returns an account's transaction history summary + individual ML risk
    score. Read-only. Design.md Section 5.
    NOTE: ml_risk_score comes from Day 6's trained model -- this is the ONE
    place ML detector output reaches the agent, exactly as Architecture.md's
    data-flow contract specifies (ml_detector -> agent: {account_id: score}).
    """
    acc_row = _accounts[_accounts["account_id"] == account_id]
    if acc_row.empty:
        return {"error": f"account_id {account_id} not found"}
    acc = acc_row.iloc[0]

    txns = _transactions[_transactions["account_id"] == account_id]
    ml_row = _ml_scores[_ml_scores["account_id"] == account_id]
    ml_score = float(ml_row.iloc[0]["ml_risk_score"]) if not ml_row.empty else None

    return {
        "account_id": account_id,
        "created_at": str(acc["created_at"]),
        "num_transactions": len(txns),
        "total_amount": float(txns["amount"].sum()),
        "avg_amount": float(txns["amount"].mean()) if len(txns) else 0.0,
        "date_range": {
            "first": str(txns["timestamp"].min()) if len(txns) else None,
            "last": str(txns["timestamp"].max()) if len(txns) else None,
        },
        "ml_risk_score": ml_score,
        "known_individual_fraud_flag": bool(txns["is_fraud_individual"].any()) if len(txns) else False,
    }


def get_connected_entities(account_id: str, max_results: int = 15) -> dict:
    """
    Returns accounts sharing device/payment/address with this account, and
    which attribute(s). Read-only. Design.md Section 5.

    CAPPED at max_results (default 15), sorted by burst_score descending --
    an account in a large real-world cluster (e.g. shared zip code) can have
    dozens of connections; returning all of them would flood the agent's
    context on a single tool call, working against the bounded-investigation
    design (Rules.md #5). Returns the most suspicious (highest burst_score)
    connections first, plus a total count so the agent knows how many exist
    beyond what's shown.
    """
    connections = _edges[
        (_edges["account_a"] == account_id) | (_edges["account_b"] == account_id)
    ]
    if connections.empty:
        return {"account_id": account_id, "total_connections": 0, "connected_accounts": []}

    results = []
    for _, row in connections.iterrows():
        other = row["account_b"] if row["account_a"] == account_id else row["account_a"]
        results.append({
            "connected_account_id": other,
            "shared_attribute": row["shared_attribute"],
            "burst_score": float(row["burst_score"]) if pd.notna(row["burst_score"]) else None,
        })

    results.sort(key=lambda r: r["burst_score"] or 0, reverse=True)
    total = len(results)
    shown = results[:max_results]

    return {
        "account_id": account_id,
        "total_connections": total,
        "connections_shown": len(shown),
        "truncated": total > max_results,
        "connected_accounts": shown,
    }


def get_similar_cases(ring_pattern_summary: dict, top_k: int = 3) -> list:
    """
    Returns top-k similar past labeled cases from the case library, with
    their outcomes. Read-only. Design.md Section 5.

    ring_pattern_summary expected shape (matches what Day 7's
    candidate_rings.json + calculate_ring_risk() already produce):
      {"size": int, "shared_attributes_involved": [str, ...],
       "ring_risk_score": float}

    Similarity is a simple, explainable heuristic (not an embedding model --
    deliberately kept transparent since this feeds the agent's evidence,
    which must itself be explainable per Rules.md #9): score = attribute
    overlap (Jaccard) + closeness in size, no black-box matching.
    """
    query_attrs = set(ring_pattern_summary.get("shared_attributes_involved", []))
    query_size = ring_pattern_summary.get("size", 0)

    scored = []
    for case in _case_library:
        case_attrs = set(case["shared_attributes"])
        if query_attrs or case_attrs:
            jaccard = len(query_attrs & case_attrs) / len(query_attrs | case_attrs) if (query_attrs | case_attrs) else 0
        else:
            jaccard = 0
        size_closeness = 1 - min(abs(query_size - case["size"]) / max(query_size, case["size"], 1), 1)
        similarity = 0.7 * jaccard + 0.3 * size_closeness
        scored.append((similarity, case))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {**case, "similarity_score": round(sim, 3)}
        for sim, case in scored[:top_k]
    ]


if __name__ == "__main__":
    # Standalone tests per Rules.md #15 -- each tool independently testable,
    # no agent loop or API call needed to sanity-check these
    print("=== get_transaction ===")
    sample_txn_id = _transactions.iloc[0]["transaction_id"]
    print(json.dumps(get_transaction(str(sample_txn_id)), indent=2, default=str))

    print("\n=== get_transaction (not found) ===")
    print(get_transaction("nonexistent_id"))

    print("\n=== get_customer_history ===")
    sample_account = json.load(open(os.path.join(DATA_DIR, "rings.json")))[0]["account_ids"][0]
    print(json.dumps(get_customer_history(sample_account), indent=2, default=str))

    print("\n=== get_connected_entities ===")
    print(json.dumps(get_connected_entities(sample_account), indent=2, default=str))

    print("\n=== get_similar_cases ===")
    candidates = json.load(open(os.path.join(DATA_DIR, "candidate_rings.json")))
    sample_candidate = candidates[0]
    query = {
        "size": sample_candidate["size"],
        "shared_attributes_involved": sample_candidate["shared_attributes_involved"],
        "ring_risk_score": sample_candidate["ring_risk_score"],
    }
    print(f"Query: {query}")
    similar = get_similar_cases(query)
    for s in similar:
        print(f"  {s['case_id']} (similarity={s['similarity_score']}): {s['outcome']} -- {s['summary'][:100]}...")

    print("\nAll 4 tools tested independently -- OK")
