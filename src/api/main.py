"""
src/api/main.py

Day 11: FastAPI wiring (PRD.md FR7; Architecture.md API layer).
Connects Day 6 (ML detector), Day 7-8 (network detection), and Day 9-10
(agent) into one callable API, per Architecture.md's data-flow contracts.

Endpoints:
  GET  /                          -- health check
  GET  /rings                     -- list all candidate rings (Day 7 output),
                                      sorted by ring_risk_score
  GET  /rings/{candidate_id}      -- detail for one candidate ring
  GET  /detect/{account_id}       -- ML risk score + basic info for one account
  POST /investigate/{candidate_id} -- run the AI Investigator agent on one
                                       candidate ring (Day 10), live Groq call

Run locally with:
  uvicorn src.api.main:app --reload

Requires GROQ_API_KEY env var set for /investigate to work (Day 10) --
/rings and /detect work without it since they only read precomputed files.
"""

import os
import sys
import json

from fastapi import FastAPI, HTTPException
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import DATA_DIR
from src.agent.investigator import investigate_ring

app = FastAPI(
    title="Sentinel Ring API",
    description="Coordinated-fraud detection: ML risk scoring + network ring "
                "detection + AI investigator recommendations.",
    version="0.1.0",
)

# Loaded once at startup -- all read-only, all precomputed by earlier stages
_candidates = json.load(open(os.path.join(DATA_DIR, "candidate_rings.json")))
_candidates_by_id = {c["candidate_id"]: c for c in _candidates}
_accounts = pd.read_csv(os.path.join(DATA_DIR, "accounts.csv"))
_ml_scores = pd.read_csv(os.path.join(DATA_DIR, "ml_detector_scores.csv"))


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "Sentinel Ring API",
        "candidates_loaded": len(_candidates),
        "accounts_loaded": len(_accounts),
    }


@app.get("/rings")
def list_rings(limit: int = 20, min_risk: float = 0.0):
    """
    List candidate rings (Day 7 output), sorted by ring_risk_score descending.
    - limit: max number of results (default 20)
    - min_risk: only return candidates with ring_risk_score >= this value
    """
    filtered = [c for c in _candidates if c["ring_risk_score"] >= min_risk]
    sorted_candidates = sorted(filtered, key=lambda c: c["ring_risk_score"], reverse=True)
    return {
        "total_matching": len(filtered),
        "returned": min(limit, len(filtered)),
        "candidates": sorted_candidates[:limit],
    }


@app.get("/rings/{candidate_id}")
def get_ring(candidate_id: str):
    """Detail for one candidate ring."""
    candidate = _candidates_by_id.get(candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail=f"candidate_id {candidate_id} not found")
    return candidate


@app.get("/detect/{account_id}")
def detect_account(account_id: str):
    """ML risk score (Day 6) + basic account info for a single account."""
    acc_row = _accounts[_accounts["account_id"] == account_id]
    if acc_row.empty:
        raise HTTPException(status_code=404, detail=f"account_id {account_id} not found")
    acc = acc_row.iloc[0]

    ml_row = _ml_scores[_ml_scores["account_id"] == account_id]
    ml_score = float(ml_row.iloc[0]["ml_risk_score"]) if not ml_row.empty else None

    # is a member of any candidate ring?
    member_of = [c["candidate_id"] for c in _candidates if account_id in c["member_accounts"]]

    return {
        "account_id": account_id,
        "ml_risk_score": ml_score,
        "is_ring_member_candidate": len(member_of) > 0,
        "candidate_ring_ids": member_of,
        "device_id": acc["device_id"],
        "created_at": str(acc["created_at"]),
    }


@app.post("/investigate/{candidate_id}")
def investigate(candidate_id: str):
    """
    Run the AI Investigator agent (Day 10) on one candidate ring.
    Makes a LIVE Groq API call -- requires GROQ_API_KEY env var set.
    This is a POST (not GET) since it costs a real API call each time,
    not just a file read like the other endpoints.
    """
    candidate = _candidates_by_id.get(candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail=f"candidate_id {candidate_id} not found")

    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY not set on the server -- cannot run live investigation.",
        )

    try:
        result = investigate_ring(candidate)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Investigation failed: {str(e)[:300]}")

    return {"candidate_id": candidate_id, "investigation": result}