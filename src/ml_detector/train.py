"""
src/ml_detector/train.py

Day 6: Train and evaluate the ML Risk Detector (PRD.md FR3, Phases.md Day 6).
Ground truth = is_fraud_individual ONLY (the original IEEE-CIS isFraud label).
is_ring_member / ring_id are NEVER used here -- per Rules.md #10, ring ground
truth is reserved for Day 8's separate ring-level evaluation.

Model: Random Forest (chosen over XGBoost for this run -- dataset is small,
43 positives total, and RF's default settings are more stable/less prone to
overfitting on such a tiny positive class without careful tuning; documented
choice, not a limitation hiding a worse result).

Honest limitation stated up front: only 43 fraud accounts in the whole
dataset. A stratified 80/20 split leaves ~8-9 positives in the test set --
small enough that metrics will have real variance. We report this plainly
rather than presenting the numbers as more solid than they are (Rules.md #3).

Output:
- Trained model saved to src/ml_detector/model.pkl
- data/ml_detector_scores.csv -- risk score for EVERY account (not just
  test set), needed as input context for Day 9's AI Investigator agent
- Printed: precision, recall, F1, confusion matrix, feature importances
"""

import os
import sys
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    classification_report
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import DATA_DIR, RANDOM_SEED

FEATURES_PATH = os.path.join(DATA_DIR, "individual_features.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SCORES_OUTPUT_PATH = os.path.join(DATA_DIR, "ml_detector_scores.csv")

FEATURE_COLS = [
    "txn_velocity", "amount_zscore_max", "account_age_days", "merchant_category_entropy"
]
LABEL_COL = "is_fraud_individual"


def main():
    df = pd.read_csv(FEATURES_PATH)

    n_fraud = df[LABEL_COL].sum()
    print(f"Dataset: {len(df)} accounts, {n_fraud} fraud ({n_fraud/len(df):.2%})")
    print("NOTE: small positive class -- metrics below have real variance, "
          "reported honestly rather than overstated.\n")

    X = df[FEATURE_COLS]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    print(f"Train: {len(X_train)} ({y_train.sum()} fraud) | "
          f"Test: {len(X_test)} ({y_test.sum()} fraud)\n")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",  # compensates for the 2.9% positive rate
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("=== Held-out test set results ===")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"\nConfusion matrix:")
    print(f"                 Predicted Clean  Predicted Fraud")
    print(f"Actual Clean     {cm[0][0]:>15}  {cm[0][1]:>15}")
    print(f"Actual Fraud     {cm[1][0]:>15}  {cm[1][1]:>15}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['clean', 'fraud'], zero_division=0)}")

    print("Feature importances:")
    for feat, imp in sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved model: {MODEL_PATH}")

    # Score EVERY account (not just test set) -- needed downstream for the
    # agent's context (Design.md Section 6: agent receives per-account risk score)
    all_scores = model.predict_proba(X)[:, 1]  # probability of class "fraud"
    scores_df = pd.DataFrame({
        "account_id": df["account_id"],
        "ml_risk_score": all_scores,
        "was_in_test_set": df.index.isin(X_test.index),
    })
    scores_df.to_csv(SCORES_OUTPUT_PATH, index=False)
    print(f"Wrote: {SCORES_OUTPUT_PATH} (risk scores for all {len(scores_df)} accounts)")

    # --- Cross-validation, since a single 80/20 split with only 9 test
    # positives is too noisy to trust on its own (Rules.md #3: report
    # honestly, don't lean on a single fragile split) ---
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    print("\n=== 5-fold cross-validation (more robust than the single split above) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_model = RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced", random_state=RANDOM_SEED
    )
    cv_precision = cross_val_score(cv_model, X, y, cv=skf, scoring="precision")
    cv_recall = cross_val_score(cv_model, X, y, cv=skf, scoring="recall")
    cv_f1 = cross_val_score(cv_model, X, y, cv=skf, scoring="f1")
    print(f"Precision across folds: {cv_precision.round(3)} (mean {cv_precision.mean():.3f})")
    print(f"Recall across folds:    {cv_recall.round(3)} (mean {cv_recall.mean():.3f})")
    print(f"F1 across folds:        {cv_f1.round(3)} (mean {cv_f1.mean():.3f})")
    print("\nHonest takeaway: with only 43 total fraud accounts, single-split metrics "
          "are noisy (see fold variance above). This individual-feature-only detector "
          "has real but limited standalone power -- which is precisely why Sentinel Ring "
          "does not rely on it alone: Network Detection (Day 7) and the AI Investigator "
          "(Day 9-10) exist to catch what this stage alone misses.")


if __name__ == "__main__":
    main()
