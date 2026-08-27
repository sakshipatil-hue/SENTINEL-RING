"""
src/features/relational.py

Day 5a: Relational feature engineering (Design.md Section 3).
Builds graph edges between accounts that share device_id, payment_token,
or billing_address_hash. Feeds Network Detection (Day 7).

IMPORTANT: placeholder/missing values must be excluded from edge creation,
or every account sharing a missing-data placeholder gets falsely linked
into one giant supercluster. Two known placeholders in this dataset:
  - device_id == "unknown_device"   (1,198 / 1,485 accounts -- most of the data)
  - billing_address_hash == "-1.0"  (67 accounts)
This is exactly the "don't silently swallow an error" principle from
Rules.md #14 -- catching this before building the graph, not after.

Output: data/relational_edges.csv
  columns: account_a, account_b, shared_attribute, attribute_value
  (one row per shared attribute per pair -- an account pair sharing 2
  attributes produces 2 rows, so edge weight = row count when aggregated
  in Day 7's graph construction)
"""

import os
import sys
from itertools import combinations

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import DATA_DIR

ACCOUNTS_PATH = os.path.join(DATA_DIR, "accounts.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "relational_edges.csv")

# Placeholder values that must NEVER be treated as a real shared identifier
PLACEHOLDER_VALUES = {
    "device_id": {"unknown_device"},
    "payment_token": set(),  # card1 has no missing values in this dataset (verified Day 2)
    "billing_address_hash": {"-1.0", "-1"},
}


def build_edges_for_attribute(accounts: pd.DataFrame, attr_col: str) -> pd.DataFrame:
    """
    For a given attribute column, group accounts by value and emit an edge
    for every pair within each group -- excluding placeholder values.
    """
    placeholders = PLACEHOLDER_VALUES.get(attr_col, set())
    values = accounts[attr_col].astype(str)
    valid = accounts[~values.isin(placeholders)].copy()
    valid["_attr_str"] = values[~values.isin(placeholders)]

    edges = []
    for value, group in valid.groupby("_attr_str"):
        if len(group) < 2:
            continue  # no pair possible, not a shared attribute
        ids = group["account_id"].tolist()
        for a, b in combinations(ids, 2):
            edges.append({
                "account_a": a,
                "account_b": b,
                "shared_attribute": attr_col,
                "attribute_value": value,
            })
    return pd.DataFrame(edges)


def main():
    accounts = pd.read_csv(ACCOUNTS_PATH)

    all_edges = []
    for attr_col in ["device_id", "payment_token", "billing_address_hash"]:
        edges = build_edges_for_attribute(accounts, attr_col)
        print(f"{attr_col}: {len(edges)} edges from shared values "
              f"(excluded placeholders: {PLACEHOLDER_VALUES.get(attr_col, set())})")
        all_edges.append(edges)

    edges_df = pd.concat(all_edges, ignore_index=True)
    edges_df.to_csv(OUTPUT_PATH, index=False)

    # Sanity check: how many edges involve at least one known ring member?
    ring_accounts = set(accounts.loc[accounts["is_ring_member"] == True, "account_id"])
    involves_ring = edges_df.apply(
        lambda r: r["account_a"] in ring_accounts or r["account_b"] in ring_accounts, axis=1
    )
    print(f"\nTotal edges: {len(edges_df)}")
    print(f"Edges involving a known ring member: {involves_ring.sum()}")
    print(f"Edges involving NO known ring member (real-world coincidental sharing): {(~involves_ring).sum()}")
    print(f"\nWrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
