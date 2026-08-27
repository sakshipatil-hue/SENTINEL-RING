"""
src/generation/inject_rings.py

Day 3: Inject 6 coordinated-fraud rings across 3 patterns (Design.md Section 2)
into the real IEEE-CIS-derived accounts/transactions data from Day 2.

Method: SELECT real accounts and RELABEL their shared-identifier columns to
create an artificial link. Never fabricate new rows from nothing (Design.md
Section 2) -- this preserves the real-noise benefit of the hybrid dataset.

Patterns:
  A) shared_device_staggered      -- shared device_id, timestamps left as-is
  B) shared_payment_burst          -- shared payment_token, timestamps compressed
                                       into a 10-20 min window
  C) address_cluster_sequential    -- shared billing_address_hash, timestamps
                                       ordered sequentially with small gaps

Output:
- data/accounts.csv       (overwritten: is_ring_member, ring_id populated)
- data/transactions.csv   (overwritten: device_id / timestamp updated for
                            ring members per pattern)
- data/rings.json         (ground truth reference table -- NEVER fed to models,
                            per Rules.md #10)
"""

import os
import sys
import json
import random
from datetime import timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import DATA_DIR, RANDOM_SEED, RING_SIZE_MIN, RING_SIZE_MAX

random.seed(RANDOM_SEED)

ACCOUNTS_PATH = os.path.join(DATA_DIR, "accounts.csv")
TRANSACTIONS_PATH = os.path.join(DATA_DIR, "transactions.csv")
RINGS_OUTPUT_PATH = os.path.join(DATA_DIR, "rings.json")

# 2 rings per pattern = 6 total, matching config.NUM_RINGS
RINGS_PER_PATTERN = 2


def load_data():
    accounts = pd.read_csv(ACCOUNTS_PATH)
    transactions = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["timestamp"])
    # ring_id loads as all-null float on first run (Day 2 wrote it empty) --
    # force to object/string dtype so string ring IDs can be assigned below
    accounts["ring_id"] = accounts["ring_id"].astype(object)
    accounts["device_id"] = accounts["device_id"].astype(object)
    accounts["payment_token"] = accounts["payment_token"].astype(object)
    accounts["billing_address_hash"] = accounts["billing_address_hash"].astype(object)
    transactions["device_id"] = transactions["device_id"].astype(object)
    return accounts, transactions


def pick_ring_accounts(accounts: pd.DataFrame, already_used: set, size: int) -> list:
    """Pick `size` accounts not already assigned to another ring."""
    candidates = accounts[~accounts["account_id"].isin(already_used)]
    chosen = candidates.sample(n=size, random_state=random.randint(0, 999999))
    return chosen["account_id"].tolist()


def inject_pattern_a(ring_id, accounts, transactions, member_ids):
    """Shared device, staggered timing -- one operator, multiple accounts over days."""
    synthetic_device = f"synthetic_device_{ring_id}"
    accounts.loc[accounts["account_id"].isin(member_ids), "device_id"] = synthetic_device
    transactions.loc[transactions["account_id"].isin(member_ids), "device_id"] = synthetic_device
    # timestamps left as-is (staggered / natural) -- this is the point of pattern A
    return {"device_id": synthetic_device}


def inject_pattern_b(ring_id, accounts, transactions, member_ids):
    """Shared payment token, burst timing -- card-testing/promo-abuse signature."""
    synthetic_token = f"synthetic_token_{ring_id}"
    accounts.loc[accounts["account_id"].isin(member_ids), "payment_token"] = synthetic_token
    transactions.loc[transactions["account_id"].isin(member_ids), "_ring_token"] = synthetic_token

    # compress timestamps into a random 10-20 min window
    burst_minutes = random.randint(10, 20)
    txn_mask = transactions["account_id"].isin(member_ids)
    n_txns = txn_mask.sum()
    base_time = transactions.loc[txn_mask, "timestamp"].min()
    offsets = sorted(random.sample(range(0, burst_minutes * 60), n_txns))
    new_times = [base_time + timedelta(seconds=off) for off in offsets]
    transactions.loc[txn_mask, "timestamp"] = new_times

    transactions.drop(columns=["_ring_token"], inplace=True, errors="ignore")
    return {"payment_token": synthetic_token, "burst_window_minutes": burst_minutes}


def inject_pattern_c(ring_id, accounts, transactions, member_ids):
    """Shared address, sequential timing -- one person operating several accounts serially."""
    synthetic_addr = f"synthetic_addr_{ring_id}"
    accounts.loc[accounts["account_id"].isin(member_ids), "billing_address_hash"] = synthetic_addr
    txn_mask = transactions["account_id"].isin(member_ids)
    n_txns = txn_mask.sum()
    base_time = transactions.loc[txn_mask, "timestamp"].min()
    # sequential, small gaps: 5-45 min apart, not simultaneous
    times = []
    t = base_time
    for _ in range(n_txns):
        t = t + timedelta(minutes=random.randint(5, 45))
        times.append(t)
    transactions.loc[txn_mask, "timestamp"] = times
    return {"billing_address_hash": synthetic_addr}


PATTERN_FUNCS = {
    "shared_device_staggered": inject_pattern_a,
    "shared_payment_burst": inject_pattern_b,
    "address_cluster_sequential": inject_pattern_c,
}


def main():
    accounts, transactions = load_data()
    used_accounts = set()
    rings_ground_truth = []
    ring_counter = 1

    for pattern_name, inject_func in PATTERN_FUNCS.items():
        for _ in range(RINGS_PER_PATTERN):
            ring_id = f"ring_{ring_counter:03d}"
            size = random.randint(RING_SIZE_MIN, RING_SIZE_MAX)
            member_ids = pick_ring_accounts(accounts, used_accounts, size)
            used_accounts.update(member_ids)

            shared_attrs = inject_func(ring_id, accounts, transactions, member_ids)

            accounts.loc[accounts["account_id"].isin(member_ids), "is_ring_member"] = True
            accounts.loc[accounts["account_id"].isin(member_ids), "ring_id"] = ring_id

            rings_ground_truth.append({
                "ring_id": ring_id,
                "pattern": pattern_name,
                "account_ids": member_ids,
                "shared_attributes": shared_attrs,
                "size": len(member_ids),
            })

            print(f"Injected {ring_id} ({pattern_name}): {len(member_ids)} accounts")
            ring_counter += 1

    accounts.to_csv(ACCOUNTS_PATH, index=False)
    transactions.to_csv(TRANSACTIONS_PATH, index=False)
    with open(RINGS_OUTPUT_PATH, "w") as f:
        json.dump(rings_ground_truth, f, indent=2, default=str)

    print(f"\nTotal rings injected: {len(rings_ground_truth)}")
    print(f"Total accounts marked as ring members: {accounts['is_ring_member'].sum()}")
    print(f"\nWrote: {ACCOUNTS_PATH}")
    print(f"Wrote: {TRANSACTIONS_PATH}")
    print(f"Wrote: {RINGS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
