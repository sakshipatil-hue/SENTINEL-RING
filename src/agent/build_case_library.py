"""
src/agent/build_case_library.py

Day 9 (part 1): Build a small labeled case library for get_similar_cases()
(Design.md Section 5, Phases.md Day 9). This is a fixed reference set the
tool searches against -- NOT a training set, NOT ground truth for evaluation
(that's rings.json). Think of it as a fraud analyst's "cases I've seen
before" notebook.

Sourced from our 6 known injected rings (ground truth) PLUS a handful of
deliberately-constructed non-ring "false alarm" cases (large innocent
clusters from candidate_rings.json that Day 8 confirmed are NOT real rings)
-- so the tool can also say "this looks like a past false alarm", not just
match against confirmed fraud. This matters for agent judgment quality:
without negative examples, get_similar_cases() could only ever push the
agent toward "yes this is fraud".

Output: data/case_library.json
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import DATA_DIR

RINGS_PATH = os.path.join(DATA_DIR, "rings.json")
CANDIDATES_PATH = os.path.join(DATA_DIR, "candidate_rings.json")
EVAL_PATH = os.path.join(DATA_DIR, "ring_evaluation.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "case_library.json")


def main():
    true_rings = json.load(open(RINGS_PATH))
    candidates = json.load(open(CANDIDATES_PATH))
    evaluation = json.load(open(EVAL_PATH))

    cases = []

    # Positive cases: our 6 confirmed injected rings
    for r in true_rings:
        cases.append({
            "case_id": f"case_{r['ring_id']}",
            "outcome": "CONFIRMED_RING",
            "pattern": r["pattern"],
            "size": len(r["account_ids"]),
            "shared_attributes": list(r["shared_attributes"].keys()),
            "summary": (
                f"Confirmed coordinated-abuse ring: {len(r['account_ids'])} accounts, "
                f"pattern={r['pattern']}, linked via {list(r['shared_attributes'].keys())}. "
                f"Action taken: escalated for review."
            ),
        })

    # Negative cases: large candidates confirmed NOT to be real rings (Day 8
    # eval), specifically single-attribute address clusters -- the exact
    # false-positive pattern Day 7/8 identified (real neighborhoods)
    confirmed_fp_ids = {
        cr["candidate_id"] for cr in evaluation["candidate_results"]
        if not cr["is_true_positive"]
    }
    candidates_by_id = {c["candidate_id"]: c for c in candidates}

    # pick a handful of large, single-attribute, high-risk-score false positives
    # -- these are the most "confusable with a real ring" cases, most useful
    # for the agent to compare against
    fp_candidates = [
        candidates_by_id[cid] for cid in confirmed_fp_ids
        if len(candidates_by_id[cid]["shared_attributes_involved"]) == 1
        and candidates_by_id[cid]["size"] >= 10
    ]
    fp_candidates.sort(key=lambda c: c["ring_risk_score"], reverse=True)

    for c in fp_candidates[:6]:  # cap at 6 negative cases -- 6 positive + 6 negative = 12 total
        cases.append({
            "case_id": f"case_{c['candidate_id']}_false_alarm",
            "outcome": "FALSE_ALARM_REAL_CLUSTER",
            "pattern": "single_attribute_cluster",
            "size": c["size"],
            "shared_attributes": c["shared_attributes_involved"],
            "summary": (
                f"Investigated cluster: {c['size']} accounts sharing only "
                f"{c['shared_attributes_involved'][0]}, ring_risk_score={c['ring_risk_score']}. "
                f"Determined to be a genuine real-world cluster (e.g. shared "
                f"neighborhood/zip code), not coordinated abuse -- single shared "
                f"attribute type, no burst timing, no cross-attribute linkage. "
                f"Action taken: no escalation, closed as false alarm."
            ),
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(cases, f, indent=2)

    print(f"Built case library: {len(cases)} cases "
          f"({sum(1 for c in cases if c['outcome']=='CONFIRMED_RING')} confirmed rings, "
          f"{sum(1 for c in cases if c['outcome']=='FALSE_ALARM_REAL_CLUSTER')} false alarms)")
    print(f"Wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
