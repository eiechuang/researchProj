import pandas as pd
import re
from difflib import SequenceMatcher



DATA_DIR = r"C:\Users\016134703\Documents\researchProj\data"

TRANSACTIONS_FILE = DATA_DIR + r"\routed_transactions.csv"
ACCOUNTS_FILE = DATA_DIR + r"\HI-Small_Accounts.csv"
OUTPUT_FILE = DATA_DIR + r"\Accounts.csv"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity(a, b):
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def fuzzy_term_flag(text, terms, threshold=0.75):
    """
    Returns 1 if the text contains or closely resembles any target term.
    Example: 'crytpo bank' can match 'crypto'.
    """
    text_clean = clean_text(text)
    words = text_clean.split()

    for term in terms:
        if term in text_clean:
            return 1

    for word in words:
        for term in terms:
            if similarity(word, term) >= threshold:
                return 1

    return 0


def find_possible_misspellings(text, target_terms, threshold=0.75):
    """
    Returns a list of possible misspellings and what they match.
    Useful for checking what the fuzzy matcher is catching.
    """
    text_clean = clean_text(text)
    words = text_clean.split()

    matches = []

    for word in words:
        for term in target_terms:
            score = similarity(word, term)

            if score >= threshold and word != term:
                matches.append({
                    "word": word,
                    "matched_term": term,
                    "score": round(score, 3)
                })

    return matches


#account cleaning

def load_accounts(accounts_file):
    accounts = pd.read_csv(accounts_file)

    accounts.columns = (accounts.columns.str.replace("\ufeff","",regex=False).str.strip())

    print("cleaned acc names") 
    print (accounts.columns.tolist())
    print() 

    accounts = accounts.rename(columns={
        "Bank Name": "bank_name",
        "Bank ID": "bank_id",
        "Account Number": "account_number",
        "Entity ID": "entity_id",
        "Entity Name": "entity_name"
    })

    print ("Columns post rename")
    print (accounts.columns.tolist())
    print ()

    required_cols = [
        "bank_name",
        "bank_id",
        "account_number",
        "entity_id",
        "entity_name"
    ]

    missing_cols = [col for col in required_cols if col not in accounts.columns]

    if missing_cols:
        raise ValueError(
            "Missing expected account column: {missing_cols}."
        )

    # Clean bank ID and account number so they match transaction file format better.
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
    accounts["entity_name"] = accounts["entity_name"].astype(str).str.strip()
    accounts["entity_id"] = accounts["entity_id"].astype(str).str.strip()
  #  accounts["bank_id"] = accounts ["bank_id"].astype(str).str.strip()

    return accounts


#bank names

CRYPTO_TERMS = [
    "crypto",
    "cryptocurrency",
    "bitcoin",
    "blockchain",
    "digital",
    "virtual"
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

ALL_TARGET_TERMS = (
    CRYPTO_TERMS
    + OFFSHORE_TERMS
    + CASINO_TERMS
    + EXCHANGE_TERMS
)


def add_bank_name_flags(accounts):
    accounts = accounts.copy()

    accounts["crypto_related_bank_flag"] = accounts["bank_name"].apply(
        lambda x: fuzzy_term_flag(x, CRYPTO_TERMS, threshold=0.75)
    )

    accounts["offshore_related_bank_flag"] = accounts["bank_name"].apply(
        lambda x: fuzzy_term_flag(x, OFFSHORE_TERMS, threshold=0.75)
    )

    accounts["casino_related_bank_flag"] = accounts["bank_name"].apply(
        lambda x: fuzzy_term_flag(x, CASINO_TERMS, threshold=0.75)
    )

    accounts["exchange_related_bank_flag"] = accounts["bank_name"].apply(
        lambda x: fuzzy_term_flag(x, EXCHANGE_TERMS, threshold=0.75)
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

    accounts["possible_bank_name_misspellings"] = accounts["bank_name"].apply(
        lambda x: find_possible_misspellings(
            x,
            ALL_TARGET_TERMS,
            threshold=0.75
        )
    )

    accounts["has_possible_bank_name_misspelling"] = (
        accounts["possible_bank_name_misspellings"].apply(len) > 0
    ).astype(int)

    return accounts
#entities

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


# -----------------------------
# Merge account info onto transactions
# -----------------------------

def load_transactions(transactions_file):
    df = pd.read_csv(transactions_file)

    print("Transaction columns:")
    print(df.columns.tolist())
    print()

    df["from_bank"] = df["from_bank"].astype(str).str.strip()
    df["to_bank"] = df["to_bank"].astype(str).str.strip()
    df["from_account"] = df["from_account"].astype(str).str.strip()
    df["to_account"] = df["to_account"].astype(str).str.strip()

    return df


def merge_accounts(transactions, accounts):
    sender_accounts = accounts.add_prefix("sender_")
    receiver_accounts = accounts.add_prefix("receiver_")

    df = transactions.merge(
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


#transactions

def add_account_transaction_features(df):
    df = df.copy()

    df["same_entity_flag"] = (
        df["sender_entity_id"] == df["receiver_entity_id"]
    ).astype(int)

    df["same_bank_name_flag"] = (
        df["sender_bank_name"] == df["receiver_bank_name"]
    ).astype(int)

    df["missing_sender_account_info"] = (
        df["sender_entity_id"].isna()
    ).astype(int)

    df["missing_receiver_account_info"] = (
        df["receiver_entity_id"].isna()
    ).astype(int)

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

    df["possible_misspelled_bank_involved_flag"] = (
        (df["sender_has_possible_bank_name_misspelling"] == 1) |
        (df["receiver_has_possible_bank_name_misspelling"] == 1)
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
# Main pipeline
# -----------------------------

def build_account_enriched_dataset(
    transactions_file=TRANSACTIONS_FILE,
    accounts_file=ACCOUNTS_FILE,
    output_file=OUTPUT_FILE
):
    print("Loading transactions...")
    transactions = load_transactions(transactions_file)

    print("Loading accounts...")
    accounts = load_accounts(accounts_file)

    print("Adding bank name flags...")
    accounts = add_bank_name_flags(accounts)

    print("Adding entity flags...")
    accounts = add_entity_flags(accounts)

    print("Merging accounts onto transactions...")
    df = merge_accounts(transactions, accounts)

    print("Adding account-level transaction features...")
    df = add_account_transaction_features(df)

    df.to_csv(output_file, index=False)

    print()
    print("Saved:", output_file)
    print("Shape:", df.shape)
    print()

    print("Missing sender account info:")
    print(df["missing_sender_account_info"].value_counts())
    print()

    print("Missing receiver account info:")
    print(df["missing_receiver_account_info"].value_counts())
    print()

    print("Crypto bank involved:")
    print(df["crypto_bank_involved_flag"].value_counts())
    print()

    print("Unusual bank involved:")
    print(df["unusual_bank_involved_flag"].value_counts())
    print()

    if "is_laundering" in df.columns:
        print("Laundering rate by unusual bank involved:")
        print(pd.crosstab(df["unusual_bank_involved_flag"], df["is_laundering"]))
        print()

        print("Laundering rate by crypto bank involved:")
        print(pd.crosstab(df["crypto_bank_involved_flag"], df["is_laundering"]))
        print()

    return df


if __name__ == "__main__":
    enriched_df = build_account_enriched_dataset()
    print(enriched_df.head())