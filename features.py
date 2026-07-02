import pandas as pd
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import sys 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



DATA_FILE = r"C:\Users\016134703\Documents\researchProj\ibm_model_ready.csv"


def evaluate_dataset(csv_path=DATA_FILE, dataset_name="IBM Bank"):
    df = pd.read_csv(csv_path)

    print("Dataset:", dataset_name)
    print("Shape:", df.shape)
    print()
    print("Class counts:")
    print(df["is_laundering"].value_counts())
    print()

    feature_cols = [
        "amount",
        "high_value_flag",
        "very_high_value_flag",
        "same_bank_flag",
        "cross_bank_flag",
        "same_account_flag",
        "currency_changed_flag",
        "amount_difference",
        "amount_ratio",
        "cross_border_flag",
        "high_risk_country_flag",
        "third_party_flag",
        "entity_obfuscation_flag",
        "rapid_movement_flag",
        "pricing_anomaly_flag",
        "risk_score",
        "payment_format",
        "typology"
    ]

    X = df[feature_cols]
    y = df["is_laundering"]

    X = pd.get_dummies(
        X,
        columns=["payment_format", "typology"],
        drop_first=False
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count

    print("scale_pos_weight:", scale_pos_weight)

    models = {
        
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )
    }

    for model_name, model in models.items():
        print("=" * 60)
        print(model_name)
        print("=" * 60)

        model.fit(X_train, y_train)

   

    y_score = model.predict_proba(X_test)[:, 1]

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)

        print("Threshold:", threshold)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
        print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))
        print("F1:", round(f1_score(y_test, y_pred, zero_division=0), 4))
        print()


if __name__ == "__main__":
    evaluate_dataset()