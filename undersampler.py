import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler
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


DATA_FILE = "data/final_ibm_bank.csv"


def load_data(data_file=DATA_FILE):
    df = pd.read_csv(data_file)

    print("Loaded:", data_file)
    print("Shape:", df.shape)
    print()
    print("Columns:")
    print(df.columns.tolist())
    print()

    return df


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
        "near_10000_flag",
        "round_1000_flag",
        "round_10000_flag",
        "large_amount_flag",
        "very_large_amount_flag",
        "cross_bank_large_flag",
        "amount_mismatch_flag",
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
        "amount_vs_sender_avg",
        "amount_vs_receiver_avg",
        "sender_unique_receivers",
        "receiver_unique_senders",
        "fan_out_flag",
        "fan_in_flag",
        "sender_daily_tx_count",
        "receiver_daily_tx_count",
        "sender_daily_total",
        "receiver_daily_total",
        "burst_sender_flag",
        "burst_receiver_flag",

        # engineered scores
        "structuring_score",
        "fan_out_score",
        "fan_in_score",
        "layering_score",
        "account_context_score",
        "engineered_risk_score",
    ]

    categorical_features = [
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


def train_xgboost(X, y, use_undersampling=True,sampling_strategy=0.01):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Before undersampling:")
    print(y_train.value_counts())
    print()

    if use_undersampling:
        under_sampler = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=42
    )

    X_train, y_train = under_sampler.fit_resample(X_train, y_train)

    print("After undersampling:")
    print(y_train.value_counts())
    print()

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count/positive_count

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
        thresholds = [0.01, 0.03, 0.05, 0.1] 

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
"""""
def show_feature_importance(model, feature_names, top_n=30):
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    ).head(top_n)

    print("=" * 60)
    print("Top Feature Importances:")
    print(importance_df)
    print()

    return importance_df
"""""

def main():
    df = load_data()

    X, y = build_feature_matrix(df)

    model, X_train, X_test, y_train, y_test, y_score = train_xgboost(X, y)

    evaluate_thresholds(y_test, y_score)


#    importance_df = show_feature_importance(
#        model,
#        X.columns,
#        top_n=40
#    )

#    importance_df.to_csv("data/SMOTEXGB_importance.csv", index=False)
#    print("Saved feature importance to data/SMOTEimportance.csv")


if __name__ == "__main__":
    main()