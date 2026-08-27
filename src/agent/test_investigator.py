"""
src/agent/test_investigator.py

Structure tests for investigator.py that don't require a live Groq API call
(the sandbox that built this project cannot reach api.groq.com -- see
investigator.py's module docstring). Tests the parts we CAN verify here:
schema validation, tool dispatch table, and the bounded-loop fallback logic
(Rules.md #5: must always return valid output, never fail silently).

Run this before attempting a live API call, to catch bugs early and cheaply.
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import DATA_DIR
from src.agent.investigator import (
    _validate_output, _build_default_output, TOOL_DISPATCH, TOOL_SCHEMAS, VALID_ACTIONS
)


def test_validate_output():
    print("=== Testing _validate_output ===")

    valid = {
        "risk_tier": "HIGH",
        "evidence": ["8 accounts share a payment token", "12 transactions in 14 minutes"],
        "recommended_action": "MANUAL_REVIEW",
        "reason": "Strong network signal but no confirmed individual fraud history.",
    }
    assert _validate_output(valid) == True, "valid output should pass"
    print("  valid output: PASS")

    missing_key = {"risk_tier": "HIGH", "evidence": ["x"], "recommended_action": "MANUAL_REVIEW"}
    assert _validate_output(missing_key) == False, "missing 'reason' should fail"
    print("  missing key rejected: PASS")

    bad_action = {**valid, "recommended_action": "AUTO_BLOCK"}
    assert _validate_output(bad_action) == False, "AUTO_BLOCK must be rejected (Rules.md #2)"
    print("  irreversible action rejected: PASS")

    bad_tier = {**valid, "risk_tier": "CRITICAL"}
    assert _validate_output(bad_tier) == False, "invalid tier should fail"
    print("  invalid risk_tier rejected: PASS")

    empty_evidence = {**valid, "evidence": []}
    assert _validate_output(empty_evidence) == False, "empty evidence should fail (Rules.md #9)"
    print("  empty evidence rejected: PASS")

    not_a_dict = "just a string"
    assert _validate_output(not_a_dict) == False, "non-dict should fail"
    print("  non-dict input rejected: PASS")

    print("All _validate_output tests PASSED\n")


def test_valid_actions_enum():
    print("=== Testing VALID_ACTIONS enum ===")
    assert VALID_ACTIONS == {"AUTO_MONITOR", "FLAG_FOR_REVIEW", "MANUAL_REVIEW"}
    assert "AUTO_BLOCK" not in VALID_ACTIONS
    assert "FREEZE" not in VALID_ACTIONS
    print("  Only non-irreversible actions present: PASS\n")


def test_default_output_fallback():
    print("=== Testing _build_default_output (call-cap fallback, Rules.md #5) ===")
    result = _build_default_output(
        "Investigation inconclusive within the 6-call budget.",
        evidence=["get_transaction(...) -> {...}"],
        tool_calls_used=6,
        hit_call_cap=True,
    )
    assert _validate_output({k: v for k, v in result.items() if k in
                              {"risk_tier", "evidence", "recommended_action", "reason"}})
    assert result["recommended_action"] == "MANUAL_REVIEW", "cap-hit fallback must default to MANUAL_REVIEW"
    assert result["hit_call_cap"] == True
    print(f"  Fallback output is schema-valid and defaults to MANUAL_REVIEW: PASS")
    print(f"  {json.dumps(result, indent=2)}\n")


def test_tool_dispatch_table():
    print("=== Testing TOOL_DISPATCH table (all 4 tools reachable, correct args) ===")
    rings = json.load(open(os.path.join(DATA_DIR, "rings.json")))
    sample_account = rings[0]["account_ids"][0]

    txn_result = TOOL_DISPATCH["get_transaction"]({"transaction_id": "2987000"})
    assert "error" not in txn_result or "transaction_id" in txn_result
    print(f"  get_transaction dispatch: OK")

    hist_result = TOOL_DISPATCH["get_customer_history"]({"account_id": sample_account})
    assert "account_id" in hist_result
    print(f"  get_customer_history dispatch: OK")

    conn_result = TOOL_DISPATCH["get_connected_entities"]({"account_id": sample_account})
    assert "total_connections" in conn_result
    print(f"  get_connected_entities dispatch: OK")

    candidates = json.load(open(os.path.join(DATA_DIR, "candidate_rings.json")))
    sim_result = TOOL_DISPATCH["get_similar_cases"]({
        "size": candidates[0]["size"],
        "shared_attributes_involved": candidates[0]["shared_attributes_involved"],
        "ring_risk_score": candidates[0]["ring_risk_score"],
    })
    assert isinstance(sim_result, list) and len(sim_result) > 0
    print(f"  get_similar_cases dispatch: OK")

    print("All TOOL_DISPATCH tests PASSED\n")


def test_tool_schemas_wellformed():
    print("=== Testing TOOL_SCHEMAS structure (Groq/OpenAI function-calling format) ===")
    assert len(TOOL_SCHEMAS) == 4, "must expose exactly the 4 tools from Design.md Section 5"
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        assert "parameters" in schema["function"]
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert names == {"get_transaction", "get_customer_history", "get_connected_entities", "get_similar_cases"}
    print(f"  All 4 tool schemas well-formed: PASS\n")


if __name__ == "__main__":
    test_validate_output()
    test_valid_actions_enum()
    test_default_output_fallback()
    test_tool_dispatch_table()
    test_tool_schemas_wellformed()
    print("=" * 50)
    print("ALL STRUCTURE TESTS PASSED")
    print("(Live Groq API call still needs to be run with a real API key")
    print(" on a machine with network access -- see investigator.py __main__)")
