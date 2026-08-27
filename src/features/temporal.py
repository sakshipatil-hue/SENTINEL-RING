"""
src/features/temporal.py

Day 5b: Temporal feature engineering (Design.md Section 3).
Computes burst_score per CONNECTED PAIR (not per account alone) -- for each
edge from relational.py, how tightly clustered in time are the two accounts'
transactions? This is what separates a real "shared_payment_burst" ring
(Pattern B, 10-20 min windows) from an innocuous coincidental match (e.g.
two strangers who both used a "Windows" device, transacting months apart).

Output: data/relational_edges.csv is REWRITTEN with an added burst_score
column (0-1, higher = more suspicious/tighter burst), so Day 7's network
detection can weight edges by both shared-attribute count AND timing.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import DATA_DIR

TRANSACTIONS_PATH = os.path.join(DATA_DIR, "transactions.csv")
EDGES_PATH = os.path.join(DATA_DIR, "relational_edges.csv")


def compute_burst_score(time_span_minutes: float) -> float:
    """
    burst_score = 1 / (time span in minutes + 1), per Design.md Section 4.
    Tighter clustering -> score closer to 1. A pair transacting a week apart
    (10,080 min) scores ~0.0001; a pair transacting 14 min apart scores ~0.067;
    same-minute transactions approach 1.0.
    """
    return 1 / (time_span_minutes + 1)


def main():
    txn = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["timestamp"])
    edges = pd.read_csv(EDGES_PATH)

    print(f"Computing burst scores for {len(edges)} edges...")

    # Precompute each account's transaction timestamps once (not per-edge query)
    account_times = txn.groupby("account_id")["timestamp"].apply(list).to_dict()

    def min_gap_minutes(acc_a, acc_b):
        times_a = account_times.get(acc_a, [])
        times_b = account_times.get(acc_b, [])
        if not times_a or not times_b:
            return None
        # minimum gap between ANY transaction of A and ANY transaction of B
        # -- captures "did these two ever transact close together", which is
        # the actual signal for a burst ring, not average gap
        min_gap = min(
            abs((ta - tb).total_seconds()) / 60
            for ta in times_a for tb in times_b
        )
        return min_gap

    edges["min_gap_minutes"] = edges.apply(
        lambda r: min_gap_minutes(r["account_a"], r["account_b"]), axis=1
    )
    edges["burst_score"] = edges["min_gap_minutes"].apply(compute_burst_score)

    edges.to_csv(EDGES_PATH, index=False)

    print(f"\nburst_score summary:")
    print(edges["burst_score"].describe())

    # Sanity check: do known ring_003/ring_004 (burst pattern) edges score high?
    print(f"\nTop 10 highest burst_score edges (should include burst-pattern ring pairs):")
    print(edges.nlargest(10, "burst_score")[["account_a", "account_b", "shared_attribute", "min_gap_minutes", "burst_score"]])

    print(f"\nUpdated: {EDGES_PATH} (added min_gap_minutes, burst_score columns)")


if __name__ == "__main__":
    main()
