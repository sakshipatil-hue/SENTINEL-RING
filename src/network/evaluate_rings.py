"""
src/network/evaluate_rings.py

Day 8: Ring-level evaluation (PRD.md success metrics; Design.md Section 7).
Grades Day 7's candidate rings against the injected ground truth (rings.json)
using the matching rule fixed in Design.md Section 7:

  - A candidate counts as a TRUE POSITIVE if it shares >= 70% of its
    members with a real injected ring (recall side: does the ring get found)
  - A real injected ring with no matching candidate is a FALSE NEGATIVE
  - A candidate with no real-ring match (>=70% overlap with ANY real ring)
    is a FALSE POSITIVE

This is a ring-level evaluation, separate and distinct from Day 6's
account-level ML evaluation (Rules.md #11: don't blend metrics from
different stages into one number that hides which stage is doing the work).

Output: printed report + data/ring_evaluation.json
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import DATA_DIR

CANDIDATES_PATH = os.path.join(DATA_DIR, "candidate_rings.json")
RINGS_GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "rings.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "ring_evaluation.json")

OVERLAP_THRESHOLD = 0.70


def main():
    candidates = json.load(open(CANDIDATES_PATH))
    true_rings = json.load(open(RINGS_GROUND_TRUTH_PATH))

    # --- Recall side: for each TRUE ring, is there a matching candidate? ---
    ring_results = []
    true_positives_recall = 0
    for r in true_rings:
        members = set(r["account_ids"])
        best_match = None
        best_overlap_pct = 0
        for c in candidates:
            overlap = set(c["member_accounts"]) & members
            pct = len(overlap) / len(members)
            if pct > best_overlap_pct:
                best_overlap_pct = pct
                best_match = c

        found = best_overlap_pct >= OVERLAP_THRESHOLD
        if found:
            true_positives_recall += 1

        ring_results.append({
            "ring_id": r["ring_id"],
            "pattern": r["pattern"],
            "true_size": len(members),
            "best_match_candidate": best_match["candidate_id"] if best_match else None,
            "best_match_rank": (candidates.index(best_match) + 1) if best_match else None,
            "overlap_pct": round(best_overlap_pct, 3),
            "found": found,
        })

    recall = true_positives_recall / len(true_rings)

    # --- Precision side: for each CANDIDATE, does it match a real ring? ---
    candidate_results = []
    true_positives_precision = 0
    for c in candidates:
        c_members = set(c["member_accounts"])
        best_overlap_pct = 0
        matched_ring = None
        for r in true_rings:
            members = set(r["account_ids"])
            overlap = c_members & members
            # precision-side overlap: fraction of the CANDIDATE that is real ring members
            pct = len(overlap) / len(c_members) if len(c_members) else 0
            if pct > best_overlap_pct:
                best_overlap_pct = pct
                matched_ring = r["ring_id"]

        is_true_positive = best_overlap_pct >= OVERLAP_THRESHOLD
        if is_true_positive:
            true_positives_precision += 1

        candidate_results.append({
            "candidate_id": c["candidate_id"],
            "size": c["size"],
            "ring_risk_score": c["ring_risk_score"],
            "matched_ring": matched_ring if is_true_positive else None,
            "overlap_pct": round(best_overlap_pct, 3),
            "is_true_positive": is_true_positive,
        })

    precision = true_positives_precision / len(candidates) if candidates else 0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    print(f"=== Ring-level evaluation (overlap threshold: {OVERLAP_THRESHOLD:.0%}) ===\n")
    print("--- Recall: did we find each real ring? ---")
    for rr in ring_results:
        status = "FOUND" if rr["found"] else "MISSED"
        print(f"  {rr['ring_id']} ({rr['pattern']}, size={rr['true_size']}): "
              f"best overlap {rr['overlap_pct']:.0%} via {rr['best_match_candidate']} "
              f"(rank #{rr['best_match_rank']}) -- {status}")

    print(f"\nRecall: {true_positives_recall}/{len(true_rings)} = {recall:.1%}")
    print(f"Precision: {true_positives_precision}/{len(candidates)} candidates = {precision:.1%}")
    print(f"F1: {f1:.3f}")

    print(f"\nNOTE: precision looks low in absolute terms because most of the "
          f"{len(candidates)} candidates are genuine real-world clusters (e.g. real "
          f"neighborhoods sharing an address) that were never meant to be flagged as "
          f"rings -- Louvain finds ALL dense communities in the graph, not just fraud "
          f"ones. This is expected: ring_risk_score (Day 7) and the AI Investigator "
          f"(Day 9-10) are what triage this candidate list down to what's worth a "
          f"human's attention, not this detection stage alone.")

    output = {
        "overlap_threshold": OVERLAP_THRESHOLD,
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "f1": round(f1, 3),
        "ring_results": ring_results,
        "candidate_results": candidate_results,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote: {OUTPUT_PATH}")

    # --- Precision @ top-K: the more realistic/useful metric for a RANKED
    # system. A reviewer wouldn't investigate all 179 candidates -- they'd
    # start from the top of the risk-sorted list. This measures whether
    # ring_risk_score is actually doing useful prioritization work. ---
    print(f"\n=== Precision @ top-K (by ring_risk_score rank) ===")
    ranked_candidates = sorted(candidates, key=lambda c: c["ring_risk_score"], reverse=True)
    for k in [5, 10, 20, 30, 50]:
        top_k = ranked_candidates[:k]
        top_k_ids = {c["candidate_id"] for c in top_k}
        hits = sum(1 for cr in candidate_results if cr["candidate_id"] in top_k_ids and cr["is_true_positive"])
        print(f"  Top {k}: {hits}/{k} are real injected rings ({hits/k:.1%})")


if __name__ == "__main__":
    main()
