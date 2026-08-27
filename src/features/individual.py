"""
src/features/individual.py

Day 4: Individual-level feature engineering (Design.md Section 3).
These features describe HOW a single account behaves -- feeds the ML Risk
Detector (Day 6). No relational/temporal-cluster features here; those are
Day 5 (src/features/relational.py, src/features/temporal.py).

Per Rules.md #10: ground truth columns (is_ring_member, ring_id) are NEVER
read here -- only is_fraud_individual is used, and only as the training
label later (Day 6), not as a feature itself.

Output: data/individual_features.csv
  columns: account_id, txn_velocity, amount_zscore_max, account_age_days,
           merchant_category_entropy, is_fraud_individual (label, kept
           separate from features at training time)
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import DATA_DIR

TRANSACTIONS_PATH = os.path.join(DATA_DIR, "transactions.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "individual_features.csv")


def compute_txn_velocity(txn: pd.DataFrame) -> pd.Series:
    """
    Transactions per account in a trailing 24h window, approximated as
    total txn count / active time span in days.

    Single-transaction accounts get velocity = 0, not a floored/inflated
    value -- a lone transaction has no "rate" to speak of, and treating it
    as high-velocity (via an artificial floor) would be a false signal.
    This matters here: ~68% of accounts in this sample have exactly 1
    transaction, so getting this right avoids corrupting the feature.
    """
    def velocity_for_group(group):
        if len(group) < 2:
            return 0.0
        span_days = (group["timestamp"].max() - group["timestamp"].min()).total_seconds() / 86400
        span_days = max(span_days, 1 / 24)  # floor at 1 hour for multi-txn bursts only
        return len(group) / span_days

    return txn.groupby("account_id").apply(velocity_for_group, include_groups=False)


def compute_amount_zscore_max(txn: pd.DataFrame) -> pd.Series:
    """
    Max absolute z-score of an account's transaction amounts vs. its OWN
    mean/std (Design.md Section 3). Accounts with only 1 transaction get 0
    (no deviation possible from a single point) -- documented, not a bug.
    """
    def zscore_for_group(group):
        if len(group) < 2:
            return 0.0
        mean = group["amount"].mean()
        std = group["amount"].std()
        if std == 0 or np.isnan(std):
            return 0.0
        z = (group["amount"] - mean).abs() / std
        return z.max()

    return txn.groupby("account_id").apply(zscore_for_group, include_groups=False)


def compute_account_age_days(txn: pd.DataFrame, reference_date=None) -> pd.Series:
    """Days since first transaction, relative to the max timestamp in the dataset
    (acts as 'now' for this synthetic-timeline dataset)."""
    if reference_date is None:
        reference_date = txn["timestamp"].max()
    first_txn = txn.groupby("account_id")["timestamp"].min()
    return (reference_date - first_txn).dt.total_seconds() / 86400


def compute_merchant_category_entropy(txn: pd.DataFrame) -> pd.Series:
    """
    Shannon entropy of an account's merchant_category distribution.
    Low entropy = account only ever uses 1 category (could be normal OR
    a signal when combined with high velocity -- the ML model decides that,
    this function just computes the raw feature).
    """
    def entropy_for_group(group):
        counts = group["merchant_category"].value_counts(normalize=True)
        return -(counts * np.log2(counts)).sum()

    return txn.groupby("account_id").apply(entropy_for_group, include_groups=False)


def compute_individual_fraud_label(txn: pd.DataFrame) -> pd.Series:
    """
    Account-level label: True if ANY of the account's transactions were
    individually flagged as fraud in the original IEEE-CIS data.
    This is the ORIGINAL is_fraud_individual signal, untouched (Rules.md #12) --
    aggregated to account level since our features are account-level.
    """
    return txn.groupby("account_id")["is_fraud_individual"].any()


def main():
    txn = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["timestamp"])

    print(f"Computing individual features for {txn['account_id'].nunique()} accounts...")

    features = pd.DataFrame({
        "txn_velocity": compute_txn_velocity(txn),
        "amount_zscore_max": compute_amount_zscore_max(txn),
        "account_age_days": compute_account_age_days(txn),
        "merchant_category_entropy": compute_merchant_category_entropy(txn),
        "is_fraud_individual": compute_individual_fraud_label(txn),
    }).reset_index().rename(columns={"index": "account_id"})

    features.to_csv(OUTPUT_PATH, index=False)

    print(f"\nFeature summary:")
    print(features.describe())
    print(f"\nFraud accounts: {features['is_fraud_individual'].sum()} / {len(features)}")
    print(f"\nWrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
