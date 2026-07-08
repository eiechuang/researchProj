import pandas as pd
import numpy as np
import re
from spellchecker import SpellChecker


DATA_DIR = r"/Users/016134703/Documents/researchproj/data/"

TRANS_FILE = DATA_DIR + r"\routed_transactions.csv"
PATTERNS_FILE = DATA_DIR + r"/routed_patterns.csv"
ACCOUNTS_FILE = DATA_DIR + r"/Accounts.csv"

OUTPUT_FILE = DATA_DIR + r"/final_ibm_bank.csv"



spell = SpellChecker(distance=1)

CUSTOM_TERMS = [
    "crypto",
    "cryptocurrency",
    "bitcoin",
    "blockchain",
    "digital",
    "virtual",
    "exchange",
    "offshore",
    "shell",
    "trust",
    "foundation",
    "holding",
    "holdings",
    "casino",
    "gambling",
    "gaming",
    "broker",
    "trading",
    "bank"
]

spell.word_frequency.load_words(CUSTOM_TERMS)


CRYPTO_TERMS = [
    "crypto",
    "cryptocurrency",
    "bitcoin",
    "blockchain"
]

OFFSHORE_TERMS = [
    "offshore",
    "shell",
    "trust",
    "foundation",
    "holding",
    "holdings"
]

CASINO_TERMS = [
    "casino",
    "gambling",
    "gaming"
]

EXCHANGE_TERMS = [
    "exchange",
    "broker",
    "trading"
]


# -----------------------------
# Text helpers
# -----------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_word(word):
    return re.sub(r"[^a-z0-9]", "", str(word).lower())


def correct_word(word):
    cleaned = clean_word(word)

    if not cleaned:
        return ""

    if cleaned in CUSTOM_TERMS:
        return cleaned

    correction = spell.correction(cleaned)

    if correction is None:
        return cleaned

    return correction


def corrected_text(text):
    words = clean_text(text).split()
    corrected_words = [correct_word(word) for word in words]
    corrected_words = [word for word in corrected_words if word]
    return " ".join(corrected_words)


def corrected_contains_term(text, terms):
    corrected = corrected_text(text)

    for term in terms:
        if term in corrected.split():
            return 1

    return 0


def was_corrected(text):
    original = clean_text(text)
    corrected = corrected_text(text)

    return int(original != corrected)


# -----------------------------
# Load and clean transactions
# -----------------------------

def load_transactions(trans_file):
    df = pd.read_csv(trans_file)

    df = df.rename(columns={
        "Timestamp": "timestamp",
        "From Bank": "from_bank",
        "Account": "from_account",
        "To Bank": "to_bank",
        "Account.1": "to_account",
        "Amount Received": "amount_received",
        "Receiving Currency": "receiving_currency",
        "Amount Paid": "amount_paid",
        "Payment Currency": "payment_currency",
        "Payment Format": "payment_format",
        "Is Laundering": "is_laundering"
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["from_bank"] = df["from_bank"].astype(str).str.strip()
    df["to_bank"] = df["to_bank"].astype(str).str.strip()
    df["from_account"] = df["from_account"].astype(str).str.strip()
    df["to_account"] = df["to_account"].astype(str).str.strip()

    df["amount_received"] = pd.to_numeric(df["amount_received"], errors="coerce").fillna(0)
    df["amount_paid"] = pd.to_numeric(df["amount_paid"], errors="coerce").fillna(0)

    df["is_laundering"] = pd.to_numeric(
        df["is_laundering"],
        errors="coerce"
    ).fillna(0).astype(int)

    return df


# -----------------------------
# Optional pattern parser
# -----------------------------

def parse_patterns(pattern_file):
    rows = []
    current_typology = None
    current_descriptor = None

    with open(pattern_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                header = line.replace("BEGIN LAUNDERING ATTEMPT - ", "")

                if ":" in header:
                    current_typology, current_descriptor = header.split(":", 1)
                    current_typology = current_typology.strip()
                    current_descriptor = current_descriptor.strip()
                else:
                    current_typology = header.strip()
                    current_descriptor = ""

                continue

            if line.startswith("END LAUNDERING ATTEMPT"):
                current_typology = None
                current_descriptor = None
                continue

            parts = line.split(",")

            if len(parts) != 11:
                continue

            rows.append({
                "timestamp": parts[0],
                "from_bank": parts[1],
                "from_account": parts[2],
                "to_bank": parts[3],
                "to_account": parts[4],
                "amount_received": float(parts[5]),
                "receiving_currency": parts[6],
                "amount_paid": float(parts[7]),
                "payment_currency": parts[8],
                "payment_format": parts[9],
                "is_laundering": int(parts[10]),
                "typology": current_typology,
                "typology_descriptor": current_descriptor
            })

    patterns_df = pd.DataFrame(rows)

    if len(patterns_df) == 0:
        return patterns_df

    patterns_df["timestamp"] = pd.to_datetime(patterns_df["timestamp"])

    return patterns_df


def merge_patterns(transactions_df, patterns_df):
    if patterns_df.empty:
        transactions_df["typology"] = "NORMAL_OR_UNLABELED"
        transactions_df["typology_descriptor"] = ""
        return transactions_df

    merge_keys = [
        "timestamp",
        "from_bank",
        "from_account",
        "to_bank",
        "to_account",
        "amount_received",
        "receiving_currency",
        "amount_paid",
        "payment_currency",
        "payment_format",
        "is_laundering"
    ]

    merged = transactions_df.merge(
        patterns_df[merge_keys + ["typology", "typology_descriptor"]],
        on=merge_keys,
        how="left"
    )

    merged["typology"] = merged["typology"].fillna("NORMAL_OR_UNLABELED")
    merged["typology_descriptor"] = merged["typology_descriptor"].fillna("")

    return merged


# -----------------------------
# Transaction feature engineering
# -----------------------------

def add_transaction_features(df):
    df = df.copy()

    df["same_bank"] = (df["from_bank"] == df["to_bank"]).astype(int)
    df["same_account"] = (df["from_account"] == df["to_account"]).astype(int)

    df["currency_changed"] = (
        df["receiving_currency"] != df["payment_currency"]
    ).astype(int)

    df["amount_difference"] = (
        df["amount_paid"] - df["amount_received"]
    ).abs()

    df["amount_ratio"] = (
        df["amount_paid"] / df["amount_received"].replace(0, pd.NA)
    ).fillna(0)

    df["log_amount_paid"] = np.log1p(df["amount_paid"])
    df["log_amount_received"] = np.log1p(df["amount_received"])

    df["near_10000_flag"] = (
        (df["amount_paid"] >= 9000) &
        (df["amount_paid"] < 10000)
    ).astype(int)

    df["round_1000_flag"] = (
        df["amount_paid"] % 1000 == 0
    ).astype(int)

    df["round_10000_flag"] = (
        df["amount_paid"] % 10000 == 0
    ).astype(int)

    df["large_amount_flag"] = (
        df["amount_paid"] >= 10000
    ).astype(int)

    df["very_large_amount_flag"] = (
        df["amount_paid"] >= 100000
    ).astype(int)

    df["cross_bank_large_flag"] = (
        (df["same_bank"] == 0) &
        (df["amount_paid"] >= 10000)
    ).astype(int)

    df["amount_mismatch_flag"] = (
        df["amount_difference"] > 0
    ).astype(int)

    df["high_amount_mismatch_flag"] = (
        (df["amount_difference"] > 0) &
        (df["amount_paid"] >= 10000)
    ).astype(int)

    return df


# -----------------------------
# Router type
# -----------------------------

def route_transaction(row):
    if row["typology"] != "NORMAL_OR_UNLABELED":
        return row["typology"]

    if row["near_10000_flag"] == 1 and row["same_bank"] == 0:
        return "BANK_STRUCTURING_LIKE"

    if row["payment_format"] == "Reinvestment":
        return "BANK_SELF_REINVESTMENT"

    if row["same_account"] == 1:
        return "BANK_SELF_TRANSFER"

    if row["currency_changed"] == 1 and row["same_bank"] == 0:
        return "BANK_CURRENCY_LAYERING_LIKE"

    if row["currency_changed"] == 1:
        return "BANK_CURRENCY_EXCHANGE"

    if row["cross_bank_large_flag"] == 1:
        return "BANK_LARGE_CROSS_BANK_TRANSFER"

    if row["same_bank"] == 0:
        return "BANK_CROSS_BANK_TRANSFER"

    if row["payment_format"] == "ACH":
        return "BANK_ACH_TRANSFER"

    if row["payment_format"] == "Wire":
        return "BANK_WIRE_TRANSFER"

    return "BANK_OTHER"


# -----------------------------
# Account loading and enrichment
# -----------------------------

def load_accounts(accounts_file):
    accounts = pd.read_csv(accounts_file)

    accounts = accounts.rename(columns={
        "Bank Name": "bank_name",
        "Bank ID": "bank_id",
        "Account Number": "account_number",
        "Entity ID": "entity_id",
        "Entity Name": "entity_name"
    })
#    required_cols = [
#        "bank_name",
#        "bank_id",
#        "account_number",
#        "entity_id",
#        "entity_name"
#    ]
    
#    missing_cols = [col for col in required_cols if col not in accounts.columns]

#    if missing_cols:
#        raise ValueError(
#            "Missing expected account column: {missing_cols}."
#        )

    accounts["bank_id"] = (
        accounts["bank_id"]
        .astype(str)
        .str.replace("#", "", regex=False)
        .str.strip()
    )

    accounts["account_number"] = (
        accounts["account_number"]
        .astype(str)
        .str.strip()
    )

    accounts["bank_name"] = accounts["bank_name"].astype(str).str.strip()
    accounts["entity_id"] = accounts["entity_id"].astype(str).str.strip()
    accounts["entity_name"] = accounts["entity_name"].astype(str).str.strip()

    return accounts


def add_bank_name_flags(accounts):
    accounts = accounts.copy()

    accounts["bank_name_corrected"] = accounts["bank_name"].apply(corrected_text)
    accounts["bank_name_was_corrected"] = accounts["bank_name"].apply(was_corrected)

    accounts["crypto_related_bank_flag"] = accounts["bank_name"].apply(
        lambda x: corrected_contains_term(x, CRYPTO_TERMS)
    )

    accounts["offshore_related_bank_flag"] = accounts["bank_name"].apply(
        lambda x: corrected_contains_term(x, OFFSHORE_TERMS)
    )

    accounts["casino_related_bank_flag"] = accounts["bank_name"].apply(
        lambda x: corrected_contains_term(x, CASINO_TERMS)
    )

    accounts["exchange_related_bank_flag"] = accounts["bank_name"].apply(
        lambda x: corrected_contains_term(x, EXCHANGE_TERMS)
    )

    accounts["unusual_bank_name_flag"] = (
        accounts[
            [
                "crypto_related_bank_flag",
                "offshore_related_bank_flag",
                "casino_related_bank_flag",
                "exchange_related_bank_flag"
            ]
        ].sum(axis=1) > 0
    ).astype(int)

    return accounts


def extract_entity_type(entity_name):
    name = clean_text(entity_name)

    if "sole proprietorship" in name or "proprietorship" in name:
        return "SOLE_PROPRIETORSHIP"

    if "llc" in name or "limited liability" in name:
        return "LLC"

    if "corporation" in name or "corp" in name:
        return "CORPORATION"

    if "limited" in name or "ltd" in name:
        return "LIMITED_COMPANY"

    if "trust" in name:
        return "TRUST"

    if "foundation" in name:
        return "FOUNDATION"

    if "partnership" in name:
        return "PARTNERSHIP"

    if "individual" in name or "person" in name:
        return "INDIVIDUAL"

    return "OTHER_ENTITY"


def add_entity_flags(accounts):
    accounts = accounts.copy()

    accounts["entity_type"] = accounts["entity_name"].apply(extract_entity_type)

    accounts["sole_proprietorship_flag"] = (
        accounts["entity_type"] == "SOLE_PROPRIETORSHIP"
    ).astype(int)

    accounts["llc_flag"] = (
        accounts["entity_type"] == "LLC"
    ).astype(int)

    accounts["corporation_flag"] = (
        accounts["entity_type"] == "CORPORATION"
    ).astype(int)

    accounts["trust_or_foundation_flag"] = (
        accounts["entity_type"].isin(["TRUST", "FOUNDATION"])
    ).astype(int)

    return accounts


def merge_accounts(df, accounts):
    sender_accounts = accounts.add_prefix("sender_")
    receiver_accounts = accounts.add_prefix("receiver_")

    df = df.merge(
        sender_accounts,
        left_on=["from_bank", "from_account"],
        right_on=["sender_bank_id", "sender_account_number"],
        how="left"
    )

    df = df.merge(
        receiver_accounts,
        left_on=["to_bank", "to_account"],
        right_on=["receiver_bank_id", "receiver_account_number"],
        how="left"
    )

    return df


def add_account_features(df):
    df = df.copy()

    df["same_entity_flag"] = (
        df["sender_entity_id"] == df["receiver_entity_id"]
    ).astype(int)

    df["same_bank_name_flag"] = (
        df["sender_bank_name"] == df["receiver_bank_name"]
    ).astype(int)

    df["missing_sender_account_info"] = df["sender_entity_id"].isna().astype(int)
    df["missing_receiver_account_info"] = df["receiver_entity_id"].isna().astype(int)

    df["crypto_bank_involved_flag"] = (
        (df["sender_crypto_related_bank_flag"] == 1) |
        (df["receiver_crypto_related_bank_flag"] == 1)
    ).astype(int)

    df["offshore_bank_involved_flag"] = (
        (df["sender_offshore_related_bank_flag"] == 1) |
        (df["receiver_offshore_related_bank_flag"] == 1)
    ).astype(int)

    df["casino_bank_involved_flag"] = (
        (df["sender_casino_related_bank_flag"] == 1) |
        (df["receiver_casino_related_bank_flag"] == 1)
    ).astype(int)

    df["exchange_bank_involved_flag"] = (
        (df["sender_exchange_related_bank_flag"] == 1) |
        (df["receiver_exchange_related_bank_flag"] == 1)
    ).astype(int)

    df["unusual_bank_involved_flag"] = (
        (df["sender_unusual_bank_name_flag"] == 1) |
        (df["receiver_unusual_bank_name_flag"] == 1)
    ).astype(int)

    df["same_entity_cross_bank_flag"] = (
        (df["same_entity_flag"] == 1) &
        (df["same_bank"] == 0)
    ).astype(int)

    df["different_entity_same_bank_flag"] = (
        (df["same_entity_flag"] == 0) &
        (df["same_bank"] == 1)
    ).astype(int)

    return df


# -----------------------------
# Behavior / graph-like features
# -----------------------------

def add_behavior_features(df):
    df = df.copy()

    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    df["sender_tx_count"] = df.groupby("from_account")["from_account"].transform("count")
    df["receiver_tx_count"] = df.groupby("to_account")["to_account"].transform("count")

    df["sender_total_amount"] = df.groupby("from_account")["amount_paid"].transform("sum")
    df["receiver_total_amount"] = df.groupby("to_account")["amount_paid"].transform("sum")

    df["sender_avg_amount"] = df.groupby("from_account")["amount_paid"].transform("mean")
    df["receiver_avg_amount"] = df.groupby("to_account")["amount_paid"].transform("mean")

    df["amount_vs_sender_avg"] = (
        df["amount_paid"] / df["sender_avg_amount"].replace(0, pd.NA)
    ).fillna(0)

    df["amount_vs_receiver_avg"] = (
        df["amount_paid"] / df["receiver_avg_amount"].replace(0, pd.NA)
    ).fillna(0)

    df["sender_unique_receivers"] = df.groupby("from_account")["to_account"].transform("nunique")
    df["receiver_unique_senders"] = df.groupby("to_account")["from_account"].transform("nunique")

    df["fan_out_flag"] = (
        df["sender_unique_receivers"] >= 5
    ).astype(int)

    df["fan_in_flag"] = (
        df["receiver_unique_senders"] >= 5
    ).astype(int)

    df["sender_daily_tx_count"] = df.groupby(["from_account", "date"])["from_account"].transform("count")
    df["receiver_daily_tx_count"] = df.groupby(["to_account", "date"])["to_account"].transform("count")

    df["sender_daily_total"] = df.groupby(["from_account", "date"])["amount_paid"].transform("sum")
    df["receiver_daily_total"] = df.groupby(["to_account", "date"])["amount_paid"].transform("sum")

    df["burst_sender_flag"] = (
        df["sender_daily_tx_count"] >= 5
    ).astype(int)

    df["burst_receiver_flag"] = (
        df["receiver_daily_tx_count"] >= 5
    ).astype(int)

    return df


# -----------------------------
# Risk score
# -----------------------------

def add_engineered_risk_score(df):
    df = df.copy()

    df["structuring_score"] = (
        df["near_10000_flag"] * 2
        + ((df["same_bank"] == 0).astype(int))
    )

    df["fan_out_score"] = (
        df["fan_out_flag"] * 2
        + df["burst_sender_flag"]
    )

    df["fan_in_score"] = (
        df["fan_in_flag"] * 2
        + df["burst_receiver_flag"]
    )

    df["layering_score"] = (
        (df["same_bank"] == 0).astype(int)
        + df["currency_changed"].astype(int)
        + df["amount_mismatch_flag"]
    )

    df["account_context_score"] = (
        df["crypto_bank_involved_flag"] * 1.5
        + df["offshore_bank_involved_flag"] * 1.5
        + df["casino_bank_involved_flag"] * 1.0
        + df["exchange_bank_involved_flag"] * 1.0
        + df["unusual_bank_involved_flag"] * 1.0
        + df["same_entity_cross_bank_flag"] * 0.75
    )

    df["engineered_risk_score"] = (
        df["structuring_score"] * 1.5
        + df["fan_out_score"] * 1.0
        + df["fan_in_score"] * 1.0
        + df["layering_score"] * 1.2
        + df["account_context_score"]
    )

    return df


# -----------------------------
# Main builder
# -----------------------------

def build_final_ibm_dataset(
    trans_file=TRANS_FILE,
    patterns_file=PATTERNS_FILE,
    accounts_file=ACCOUNTS_FILE,
    output_file=OUTPUT_FILE
):
    print("Loading transactions...")
    df = load_transactions(trans_file)

    print("Adding transaction features...")
    df = add_transaction_features(df)

    print("Parsing patterns...")
    patterns_df = parse_patterns(patterns_file)

    print("Merging pattern typologies...")
    df = merge_patterns(df, patterns_df)

    print("Assigning router types...")
    df["router_type"] = df.apply(route_transaction, axis=1)

    print("Loading accounts...")
    accounts = load_accounts(accounts_file)

    print("Adding account flags...")
    accounts = add_bank_name_flags(accounts)
    accounts = add_entity_flags(accounts)

    print("Merging account data...")
    df = merge_accounts(df, accounts)

    print("Adding account features...")
    df = add_account_features(df)

    print("Adding behavior features...")
    df = add_behavior_features(df)

    print("Adding engineered risk score...")
    df = add_engineered_risk_score(df)

    df.to_csv(output_file, index=False)

    print()
    print("Saved:", output_file)
    print("Shape:", df.shape)
    print()

    print("Class counts:")
    print(df["is_laundering"].value_counts())
    print()

    print("Router type counts:")
    print(df["router_type"].value_counts().head(20))
    print()

    print("Crypto bank involved:")
    print(df["crypto_bank_involved_flag"].value_counts())
    print()

    print("Unusual bank involved:")
    print(df["unusual_bank_involved_flag"].value_counts())
    print()

    return df


if __name__ == "__main__":
    final_df = build_final_ibm_dataset()
    print(final_df.head())