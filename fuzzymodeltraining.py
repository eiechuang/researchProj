import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


DATA_FILE = "data/fuzzy_final_ibm_bank.csv"


def load_data(data_file=DATA_FILE):
    df = pd.read_csv(data_file)

    print("Loaded:", data_file)
    print("Shape:", df.shape)
    print()
    print("Columns:")
    print(df.columns.tolist())
    print()

    return df

#  fuzzy_fan_out, fuzzy fan in , fuzzy_sender_burst, fuzzy reciever burst, fuzzy amount vs sender avg, fuzzy amount vs reciever avg. 
#rplace: fuzzy_near_10000, fuzzy_round_1000, fuzzy_round_10000, fuzzy_large_amount, fuzzy_very_large_amount, fuzzy_amount mismatch, fuzzy_fan_out
def build_feature_matrix(df):
    target_col = "is_laundering"

    numeric_features = [
        # original transaction features
        "amount_paid",
        "amount_received",
        "same_bank",
        "same_account",
        "currency_changed",
        "amount_difference",
        "amount_ratio",

        # amount features
        "log_amount_paid",
        "log_amount_received",
        "fuzzy_near_10000", #needs replacing -> 
        "fuzzy_round_1000",
        "fuzzy_round_10000",
        "fuzzy_large_amount",
        "fuzzy_very_large_amount",
        "cross_bank_large_flag",
        "fuzzy_amount mismatch",
        "high_amount_mismatch_flag",

        # account/entity features
        "same_entity_flag",
        "same_bank_name_flag",
        "missing_sender_account_info",
        "missing_receiver_account_info",
        "crypto_bank_involved_flag",
        "offshore_bank_involved_flag",
        "casino_bank_involved_flag",
        "exchange_bank_involved_flag",
        "unusual_bank_involved_flag",
        "same_entity_cross_bank_flag",
        "different_entity_same_bank_flag",

        # sender entity flags
        "sender_sole_proprietorship_flag",
        "sender_llc_flag",
        "sender_corporation_flag",
        "sender_trust_or_foundation_flag",

        # receiver entity flags
        "receiver_sole_proprietorship_flag",
        "receiver_llc_flag",
        "receiver_corporation_flag",
        "receiver_trust_or_foundation_flag",

        # behavior / graph-like features
        "sender_tx_count",
        "receiver_tx_count",
        "sender_total_amount",
        "receiver_total_amount",
        "sender_avg_amount",
        "receiver_avg_amount",
        "fuzzy_amount_vs_sender_avg",
        "fuzzy_amount_vs_reciever_avg",
        "sender_unique_receivers",
        "receiver_unique_senders",
        "fuzzy_fan_out", #replacement needed
        "fuzzy_fan_in",
        "sender_daily_tx_count",
        "receiver_daily_tx_count",
        "sender_daily_total",
        "receiver_daily_total",
        "fuzzy_sender_burst",
        "fuzzy_reciever_burst",

        # engineered scores
        "structuring_score",
        "fan_out_score",
        "fan_in_score",
        "layering_score",
        "account_context_score",
        "engineered_risk_score",
    ]

    categorical_features = [ #needs replacement
        "payment_format",
        "router_type",
        "sender_entity_type",
        "receiver_entity_type",
    ]

    # Keep only columns that actually exist, so the script does not crash
    numeric_features = [col for col in numeric_features if col in df.columns]
    categorical_features = [col for col in categorical_features if col in df.columns]

    print("Using numeric features:", len(numeric_features))
    print(numeric_features)
    print()

    print("Using categorical features:", len(categorical_features))
    print(categorical_features)
    print()

    X_num = df[numeric_features].copy()

    for col in X_num.columns:
        X_num[col] = pd.to_numeric(X_num[col], errors="coerce").fillna(0)

    X_cat = pd.get_dummies(
        df[categorical_features].fillna("UNKNOWN").astype(str),
        columns=categorical_features,
        drop_first=False
    )

    X = pd.concat([X_num, X_cat], axis=1)

    y = df[target_col].astype(int)

    print("Final feature matrix shape:", X.shape)
    print("Target counts:")
    print(y.value_counts())
    print()

    return X, y

"""""""""
def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count / positive_count

    print("Training counts:")
    print("Negative:", negative_count)
    print("Positive:", positive_count)
    print("scale_pos_weight:", scale_pos_weight)
    print()

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)[:, 1]

    return model, X_train, X_test, y_train, y_test, y_score
"""""""""

def train_xgboost(X, y, use_smote=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Before SMOTE:")
    print(y_train.value_counts())
    print()

    if use_smote:
        smote = SMOTE(
            sampling_strategy=0.2,  # minority becomes 10% of majority
            random_state=42,
            k_neighbors=5
        )

        X_train, y_train = smote.fit_resample(X_train, y_train)

        print("After SMOTE:")
        print(y_train.value_counts())
        print()

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = positive_count/negative_count

    print("Training counts:")
    print("Negative:", negative_count)
    print("Positive:", positive_count)
    print("scale_pos_weight:", scale_pos_weight)
    print()

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)[:, 1]

    return model, X_train, X_test, y_train, y_test, y_score

def evaluate_thresholds(y_test, y_score, thresholds=None):
    if thresholds is None:
        thresholds = [0.3, 0.35]

    print("ROC-AUC:", round(roc_auc_score(y_test, y_score), 4))
    print("PR-AUC:", round(average_precision_score(y_test, y_score), 4))
    print()

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)

        cm = confusion_matrix(y_test, y_pred)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print("=" * 60)
        print("Threshold:", threshold)
        print("Confusion Matrix:")
        print(cm)
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1:", round(f1, 4))
        print()

def evaluate_at_fp_budget(y_test, y_score, fp_budget=100000):
    results = pd.DataFrame({
        "actual": y_test.values,
        "score": y_score
    })

    results = results.sort_values("score", ascending=False).reset_index(drop=True)

    results["false_positive"] = (results["actual"] == 0).astype(int)
    results["true_positive"] = (results["actual"] == 1).astype(int)

    results["cum_fp"] = results["false_positive"].cumsum()
    results["cum_tp"] = results["true_positive"].cumsum()

    allowed = results[results["cum_fp"] <= fp_budget]

    tp = allowed["true_positive"].sum()
    fp = allowed["false_positive"].sum()
    flagged = len(allowed)
    total_pos = results["actual"].sum()

    recall = tp / total_pos if total_pos > 0 else 0
    precision = tp / flagged if flagged > 0 else 0

    print("=" * 60)
    print("False-positive budget:", fp_budget)
    print("Flagged transactions:", flagged)
    print("True positives:", int(tp))
    print("False positives:", int(fp))
    print("Recall:", round(recall, 4))
    print("Precision:", round(precision, 4))
    print()


def show_feature_importance(model, feature_names, top_n=30):
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    ).head(top_n)

#    print("=" * 60)
 #   print("Top Feature Importances:")
  #  print(importance_df)
 #   print()

    return importance_df
def find_threshold_for_recall(y_test, y_score, target_recall=0.96):
    results = pd.DataFrame({
        "actual": y_test.values,
        "score": y_score
    })

    results = results.sort_values("score", ascending=False).reset_index(drop=True)

    results["true_positive"] = (results["actual"] == 1).astype(int)
    results["false_positive"] = (results["actual"] == 0).astype(int)

    results["cum_tp"] = results["true_positive"].cumsum()
    results["cum_fp"] = results["false_positive"].cumsum()

    total_laundering = results["actual"].sum()
    results["recall"] = results["cum_tp"] / total_laundering
    results["precision"] = results["cum_tp"] / (results.index + 1)

    candidates = results[results["recall"] >= target_recall]

    if candidates.empty:
        print("No threshold reaches recall:", target_recall)
        return None

    best = candidates.iloc[0]

    threshold = best["score"]
    true_positives = int(best["cum_tp"])
    false_positives = int(best["cum_fp"])
    flagged = int(best.name + 1)
    recall = best["recall"]
    precision = best["precision"]

    print("=" * 60)
    print("Target recall:", target_recall)
    print("Threshold needed:", threshold)
    print("Flagged transactions:", flagged)
    print("True positives:", true_positives)
    print("False positives:", false_positives)
    print("Recall:", round(recall, 4))
    print("Precision:", round(precision, 6))
    print()

    return threshold

def evaluate_at_fp_budget(y_test, y_score, fp_budget=100000):
    results = pd.DataFrame({
        "actual": y_test.values,
        "score": y_score
    })

    results = results.sort_values("score", ascending=False).reset_index(drop=True)

    results["true_positive"] = (results["actual"] == 1).astype(int)
    results["false_positive"] = (results["actual"] == 0).astype(int)

    results["cum_tp"] = results["true_positive"].cumsum()
    results["cum_fp"] = results["false_positive"].cumsum()

    allowed = results[results["cum_fp"] <= fp_budget]

    if allowed.empty:
        print("No transactions allowed under FP budget:", fp_budget)
        return

    true_positives = int(allowed["true_positive"].sum())
    false_positives = int(allowed["false_positive"].sum())
    flagged = len(allowed)

    total_laundering = results["actual"].sum()
    recall = true_positives / total_laundering
    precision = true_positives / flagged

    threshold = allowed.iloc[-1]["score"]

    print("=" * 60)
    print("False-positive budget:", fp_budget)
    print("Threshold at budget:", threshold)
    print("Flagged transactions:", flagged)
    print("True positives:", true_positives)
    print("False positives:", false_positives)
    print("Recall:", round(recall, 4))
    print("Precision:", round(precision, 6))
    print()

def main():
    df = load_data()

    X, y = build_feature_matrix(df)

    model, X_train, X_test, y_train, y_test, y_score = train_xgboost(X, y)

    evaluate_thresholds(y_test, y_score)

    print("=" * 60)
    print("Thresholds required for target recall")
    print("=" * 60)

    threshold_96 = find_threshold_for_recall(
        y_test,
        y_score,
        target_recall=0.96
    )

    threshold_98 = find_threshold_for_recall(
        y_test,
        y_score,
        target_recall=0.98
    )

    print("=" * 60)
    print("Recall at false-positive budgets")
    print("=" * 60)

    for budget in [25000, 50000, 75000, 100000, 125000, 150000]:
        evaluate_at_fp_budget(
            y_test,
            y_score,
            fp_budget=budget
        )


if __name__ == "__main__":
    main()