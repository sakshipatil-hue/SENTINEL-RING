"""
src/agent/investigator.py

Day 10: The AI Investigator agent loop (Design.md Section 5-6, Architecture.md).
Wires the 4 tools from Day 9 into a bounded (max 6 calls) tool-use loop
using the Groq API (Llama 3.3 70B) -- an OpenAI-compatible function-calling
format.

WHY GROQ (not the originally-planned Claude API): switched after the team
confirmed a $0 budget constraint (Memory.md 2026-08-22 entry). Groq's free
tier supports reliable tool-calling with a fast, capable open model.
Model: openai/gpt-oss-120b (Groq deprecated the earlier llama-3.3-70b-versatile
model; gpt-oss-120b is Groq's current recommended model for tool-calling /
reasoning workloads as of this build -- if Groq deprecates this one too,
check console.groq.com/docs/models for the current recommendation and
update the `model` default above).

NETWORK NOTE: this file was written and structure-tested in a sandboxed
build environment whose network allowlist does not include api.groq.com --
so the tool-dispatch logic, prompt construction, and output-schema
validation were tested directly (see test_investigator.py), but a live
end-to-end API call must be run on a machine with real internet access and
a Groq API key (free at console.groq.com/keys).

Enforces (Rules.md):
  #5  bounded loop, max 6 tool calls, always returns valid structured output
      even if the cap is hit before a decision
  #6  calculate_ring_risk() is NEVER agent-callable -- it's precomputed
      (Day 7) and handed to the agent as fixed context
  #7  all 4 tools are read-only
  #8  final output must validate against the schema -- free text is fine as
      internal reasoning, never as the final answer
  #9  every recommendation must cite evidence
  #2  recommended_action is restricted to AUTO_MONITOR / FLAG_FOR_REVIEW /
      MANUAL_REVIEW -- never an irreversible action
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import DATA_DIR, MAX_AGENT_TOOL_CALLS
from src.agent.tools import (
    get_transaction, get_customer_history, get_connected_entities, get_similar_cases
)

VALID_ACTIONS = {"AUTO_MONITOR", "FLAG_FOR_REVIEW", "MANUAL_REVIEW"}

# --- Tool schema definitions (OpenAI/Groq function-calling format) ---
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_transaction",
            "description": "Get full details of a single transaction by ID.",
            "parameters": {
                "type": "object",
                "properties": {"transaction_id": {"type": "string"}},
                "required": ["transaction_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer_history",
            "description": "Get an account's transaction history summary and its individual ML fraud risk score.",
            "parameters": {
                "type": "object",
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_connected_entities",
            "description": "Get accounts connected to this one via shared device, payment instrument, or address (top 15 most suspicious, by burst timing).",
            "parameters": {
                "type": "object",
                "properties": {"account_id": {"type": "string"}},
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_similar_cases",
            "description": "Find similar past investigated cases (confirmed rings or false alarms) to compare this candidate against.",
            "parameters": {
                "type": "object",
                "properties": {
                    "size": {"type": "integer"},
                    "shared_attributes_involved": {"type": "array", "items": {"type": "string"}},
                    "ring_risk_score": {"type": "number"},
                },
                "required": ["size", "shared_attributes_involved", "ring_risk_score"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "get_transaction": lambda args: get_transaction(args["transaction_id"]),
    "get_customer_history": lambda args: get_customer_history(args["account_id"]),
    "get_connected_entities": lambda args: get_connected_entities(args["account_id"]),
    "get_similar_cases": lambda args: get_similar_cases({
        "size": args.get("size"),
        "shared_attributes_involved": args.get("shared_attributes_involved", []),
        "ring_risk_score": args.get("ring_risk_score"),
    }),
}

SYSTEM_PROMPT = f"""You are a fraud investigation agent for Sentinel Ring, a coordinated-abuse detection system.

GOAL: investigate ONE candidate ring (a group of accounts flagged as potentially coordinating fraud) and produce a structured recommendation for a human reviewer.

YOU ARE GIVEN, as fixed context (never recompute these yourself):
- The candidate ring's member account IDs
- A deterministic ring_risk_score (0-1) already computed from size, density, shared-attribute diversity, and timing -- this was computed upstream, treat it as ground truth input, do not question or recompute it

YOU HAVE 4 READ-ONLY TOOLS to gather further evidence:
- get_transaction(transaction_id)
- get_customer_history(account_id)
- get_connected_entities(account_id)
- get_similar_cases(size, shared_attributes_involved, ring_risk_score)

RULES YOU MUST FOLLOW:
1. You have a HARD LIMIT of {MAX_AGENT_TOOL_CALLS} tool calls total. Use them wisely -- don't call the same tool redundantly. Budget guidance: for a ring with several members, do NOT call get_connected_entities on every single member -- check 1-2 representative accounts, then use get_similar_cases and get_customer_history to round out your evidence. Leave at least 1 call of margin before the limit so you have room to reach a final decision.
2. You NEVER recommend an irreversible action. Your recommended_action MUST be exactly one of: AUTO_MONITOR, FLAG_FOR_REVIEW, MANUAL_REVIEW. Never "block", "freeze", "reject", or similar.
3. Every recommendation MUST be backed by specific evidence you gathered -- cite account counts, shared attributes, timing patterns, similar past cases. No evidence, no recommendation.
4. If you are uncertain or the evidence is mixed, prefer MANUAL_REVIEW over guessing.
5. As soon as you have enough evidence to make a confident call -- often after just 2-4 tool calls -- STOP calling tools and respond with your final answer as PLAIN TEXT containing ONLY a JSON object in the schema below. Do NOT attempt to call a function/tool named "json" or anything similar to produce this output -- write it directly as your text response, not as a tool call.

{{
  "risk_tier": "HIGH" | "MEDIUM" | "LOW",
  "evidence": ["evidence point 1", "evidence point 2", ...],
  "recommended_action": "AUTO_MONITOR" | "FLAG_FOR_REVIEW" | "MANUAL_REVIEW",
  "reason": "one to two sentence explanation connecting the evidence to the recommendation"
}}
"""


def _build_default_output(reason: str, evidence: list, tool_calls_used: int, hit_call_cap: bool) -> dict:
    """Fallback structured output -- used when the agent hits the call cap
    without deciding, or when its final response fails schema validation.
    Per Rules.md #5: the loop must NEVER silently fail or hang; it always
    returns valid structured output."""
    return {
        "risk_tier": "MEDIUM",
        "evidence": evidence or ["Investigation did not complete with sufficient evidence."],
        "recommended_action": "MANUAL_REVIEW",
        "reason": reason,
        "tool_calls_used": tool_calls_used,
        "hit_call_cap": hit_call_cap,
    }


def _validate_output(parsed: dict) -> bool:
    """Rules.md #8: final output must validate against the schema."""
    if not isinstance(parsed, dict):
        return False
    required_keys = {"risk_tier", "evidence", "recommended_action", "reason"}
    if not required_keys.issubset(parsed.keys()):
        return False
    if parsed["risk_tier"] not in {"HIGH", "MEDIUM", "LOW"}:
        return False
    if parsed["recommended_action"] not in VALID_ACTIONS:
        return False
    if not isinstance(parsed["evidence"], list) or len(parsed["evidence"]) == 0:
        return False
    return True


def investigate_ring(candidate: dict, model: str = "openai/gpt-oss-120b", api_key: str = None) -> dict:
    """
    Run the bounded agent loop on a single candidate ring.

    candidate: one entry from data/candidate_rings.json, e.g.
      {"candidate_id": ..., "member_accounts": [...], "size": ...,
       "ring_risk_score": ..., "shared_attributes_involved": [...]}

    Requires GROQ_API_KEY env var (or api_key param) -- get a free key at
    console.groq.com/keys. This function makes a real network call and
    cannot be tested inside the sandboxed build environment (see module
    docstring) -- run this on a machine with internet access.
    """
    from groq import Groq

    client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    user_context = f"""Investigate this candidate ring:
- candidate_id: {candidate['candidate_id']}
- member_accounts: {candidate['member_accounts']}
- size: {candidate['size']}
- ring_risk_score (precomputed, do not recompute): {candidate['ring_risk_score']}
- shared_attributes_involved: {candidate['shared_attributes_involved']}

Investigate and produce your structured recommendation."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_context},
    ]

    tool_calls_used = 0
    evidence_gathered = []

    while tool_calls_used < MAX_AGENT_TOOL_CALLS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.2,
            )
        except Exception as e:
            # Known quirk with some Groq/open models: instead of returning the
            # final JSON as plain text, the model sometimes "calls" a
            # non-existent pseudo-tool named "json" to emit structured output.
            # Groq's API rejects this as an invalid tool call (400 error), but
            # the actual answer is present in the error's failed_generation
            # field -- extract and use it rather than treating this as a hard
            # failure (Rules.md #5: never fail silently, always return valid
            # structured output).
            err_str = str(e)
            if "attempted to call tool 'json'" in err_str:
                try:
                    # Locate the "arguments": { ... } object using balanced-
                    # brace matching (robust to exact quote/escape formatting
                    # differences in how the SDK renders the error string --
                    # more reliable than a fixed string-slice).
                    marker = '"arguments":'
                    arg_start = err_str.index(marker) + len(marker)
                    # skip whitespace to the opening brace
                    while err_str[arg_start] in " \n\t":
                        arg_start += 1
                    assert err_str[arg_start] == "{"
                    depth = 0
                    i = arg_start
                    while i < len(err_str):
                        if err_str[i] == "{":
                            depth += 1
                        elif err_str[i] == "}":
                            depth -= 1
                            if depth == 0:
                                break
                        i += 1
                    raw_obj = err_str[arg_start:i + 1]
                    raw_obj = raw_obj.encode().decode("unicode_escape")
                    parsed = json.loads(raw_obj)
                    if _validate_output(parsed):
                        parsed["tool_calls_used"] = tool_calls_used
                        parsed["hit_call_cap"] = False
                        parsed["note"] = "Recovered from model's non-standard 'json' pseudo-tool-call quirk."
                        return parsed
                except Exception:
                    pass  # fall through to generic failure handling below
            return _build_default_output(
                f"API call failed: {err_str[:200]}",
                evidence_gathered, tool_calls_used, hit_call_cap=False,
            )

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})
            for tc in msg.tool_calls:
                if tool_calls_used >= MAX_AGENT_TOOL_CALLS:
                    break
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                if fn_name not in TOOL_DISPATCH:
                    result = {"error": f"unknown tool {fn_name}"}
                else:
                    result = TOOL_DISPATCH[fn_name](fn_args)
                tool_calls_used += 1
                evidence_gathered.append(f"{fn_name}({fn_args}) -> {json.dumps(result, default=str)[:200]}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                })
            continue

        # No more tool calls -- agent should have produced final JSON
        try:
            content = msg.content.strip()
            if content.startswith("```"):
                content = content.strip("`").removeprefix("json").strip()
            parsed = json.loads(content)
        except (json.JSONDecodeError, AttributeError):
            return _build_default_output(
                "Agent response could not be parsed as valid JSON.",
                evidence_gathered, tool_calls_used, hit_call_cap=False,
            )

        if not _validate_output(parsed):
            return _build_default_output(
                "Agent response did not match required schema.",
                evidence_gathered, tool_calls_used, hit_call_cap=False,
            )

        parsed["tool_calls_used"] = tool_calls_used
        parsed["hit_call_cap"] = False
        return parsed

    # Hit the call cap without a decision (Rules.md #5)
    return _build_default_output(
        f"Investigation inconclusive within the {MAX_AGENT_TOOL_CALLS}-call budget.",
        evidence_gathered, tool_calls_used, hit_call_cap=True,
    )


if __name__ == "__main__":
    candidates = json.load(open(os.path.join(DATA_DIR, "candidate_rings.json")))
    top_candidate = candidates[0]
    print(f"Investigating {top_candidate['candidate_id']} (risk_score={top_candidate['ring_risk_score']})...")
    result = investigate_ring(top_candidate)
    print(json.dumps(result, indent=2))

