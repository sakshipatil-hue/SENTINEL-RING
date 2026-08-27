"""
src/generation/load_and_map.py

Day 2: Load raw IEEE-CIS data and map it to our schema (Design.md Section 1).
No ring injection yet — that's Day 3 (src/generation/inject_rings.py).

Output:
- data/transactions.csv  (schema: transaction_id, account_id, timestamp, amount,
                           transaction_type, merchant_category, device_id,
                           is_fraud_individual)
- data/accounts.csv      (schema: account_id, created_at, device_id, ip_subnet,
                           payment_token, billing_address_hash,
                           is_ring_member, ring_id)

Ground truth columns (is_ring_member, ring_id) are created here as placeholders
(False / null) — Day 3 will populate them. Per Rules.md #10, these must NEVER
be fed into model features later.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import DATA_DIR, RANDOM_SEED

RAW_TXN_PATH = os.path.join(DATA_DIR, "raw_transactions.csv")
RAW_IDENTITY_PATH = os.path.join(DATA_DIR, "raw_identity.csv")

# Anchor date: IEEE-CIS's TransactionDT is "seconds since a reference point",
# not a real calendar date. We anchor it to an arbitrary recent date so our
# timestamps are realistic and usable for temporal features (Design.md Section 3).
ANCHOR_DATE = datetime(2026, 6, 1)


def load_raw():
    txn = pd.read_csv(RAW_TXN_PATH)
    identity = pd.read_csv(RAW_IDENTITY_PATH)
    print(f"Loaded {len(txn)} transactions, {len(identity)} identity records")
    return txn, identity


def map_transactions(txn: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
    """Map raw IEEE-CIS columns to our transactions schema (Design.md Section 1)."""
    df = txn.merge(
        identity[["TransactionID", "DeviceInfo"]], on="TransactionID", how="left"
    )

    mapped = pd.DataFrame()
    mapped["transaction_id"] = df["TransactionID"].astype(str)
    # account_id derived below in derive_accounts(); placeholder for now
    mapped["_card1"] = df["card1"]
    mapped["_addr1"] = df["addr1"]

    # TransactionDT is seconds-from-reference -> convert to real datetime
    mapped["timestamp"] = df["TransactionDT"].apply(
        lambda secs: ANCHOR_DATE + timedelta(seconds=int(secs))
    )
    mapped["amount"] = df["TransactionAmt"].astype(float)

    # transaction_type: IEEE-CIS doesn't have purchase/return/refund directly.
    # ProductCD is the closest proxy we have (documented approximation,
    # per Design.md Section 1 note on merchant_category).
    mapped["transaction_type"] = "purchase"  # IEEE-CIS is all purchase-side data;
    # returns/refunds aren't distinguished in this dataset. Documented limitation.

    mapped["merchant_category"] = df["ProductCD"].fillna("unknown")
    mapped["device_id"] = df["DeviceInfo"].fillna("unknown_device")

    # Preserve original ground truth EXACTLY as-is (Rules.md #12 — never modify)
    mapped["is_fraud_individual"] = df["isFraud"].astype(bool)

    return mapped


def derive_accounts(mapped_txn: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Derive accounts from unique (card1, addr1) combinations (Design.md Section 1
    account derivation rule). Returns (accounts_df, transactions_df_with_account_id).
    """
    # Handle missing addr1 -> treat as its own bucket per card, not silently dropped
    mapped_txn["_addr1_filled"] = mapped_txn["_addr1"].fillna(-1)

    key_cols = ["_card1", "_addr1_filled"]
    unique_keys = mapped_txn[key_cols].drop_duplicates().reset_index(drop=True)
    unique_keys["account_id"] = "acc_" + unique_keys.index.astype(str).str.zfill(5)

    merged = mapped_txn.merge(unique_keys, on=key_cols, how="left")

    # accounts table: one row per derived account
    acc_group = merged.groupby("account_id").agg(
        payment_token=("_card1", "first"),
        billing_address_hash=("_addr1_filled", "first"),
        device_id=("device_id", "first"),  # first-seen device for this account
        created_at=("timestamp", "min"),   # earliest transaction = account "creation"
    ).reset_index()

    acc_group["ip_subnet"] = "unknown"  # IEEE-CIS has no IP field; documented gap
    acc_group["is_ring_member"] = False  # placeholder, Day 3 populates
    acc_group["ring_id"] = None          # placeholder, Day 3 populates

    accounts_df = acc_group[[
        "account_id", "created_at", "device_id", "ip_subnet",
        "payment_token", "billing_address_hash", "is_ring_member", "ring_id"
    ]]

    transactions_df = merged[[
        "transaction_id", "account_id", "timestamp", "amount",
        "transaction_type", "merchant_category", "device_id",
        "is_fraud_individual"
    ]]

    return accounts_df, transactions_df


def main():
    txn, identity = load_raw()
    mapped = map_transactions(txn, identity)
    accounts_df, transactions_df = derive_accounts(mapped)

    accounts_path = os.path.join(DATA_DIR, "accounts.csv")
    transactions_path = os.path.join(DATA_DIR, "transactions.csv")
    accounts_df.to_csv(accounts_path, index=False)
    transactions_df.to_csv(transactions_path, index=False)

    print(f"\nDerived {len(accounts_df)} unique accounts from {len(transactions_df)} transactions")
    print(f"Fraud rate: {transactions_df['is_fraud_individual'].mean():.3%}")
    print(f"Accounts with device info: {(accounts_df['device_id'] != 'unknown_device').sum()} / {len(accounts_df)}")
    print(f"\nWrote: {accounts_path}")
    print(f"Wrote: {transactions_path}")


if __name__ == "__main__":
    main()
