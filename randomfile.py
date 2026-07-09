import pandas as pd
DATA_DIR =  r"C:\Users\016134703\Documents\researchProj\data"

TRANSACTIONS_CSV = DATA_DIR + r"\routed_transactions.csv"
PATTERNS_CSV = DATA_DIR + r"\routed_patterns.csv"
ACCOUNTS_CSV = DATA_DIR + r"\routed_accounts"

OUTPUT_CSV = DATA_DIR + "IBMBank.csv"


def load_files(transactions_csv, patterns_csv, ):
    transactions_df = pd.read_csv(transactions_csv)
    patterns_df = pd.read_csv(patterns_csv)
   # accounts_df = pad.read_csv(accounts_csv)  #read accounts dataframe, need to ascertain accounts_csv reference

    transactions_df["timestamp"] = pd.to_datetime(transactions_df["timestamp"])
    patterns_df["timestamp"] = pd.to_datetime(patterns_df["timestamp"])


    return transactions_df, patterns_df


def add_router_features(df):
    df = df.copy()

    df["same_bank"] = (df["from_bank"] == df["to_bank"]).astype(int)
    df["same_account"] = (df["from_account"] == df["to_account"]).astype(int)

    df["currency_changed"] = (
        df["receiving_currency"] != df["payment_currency"]
    ).astype(int)

    df["amount_difference"] = (
        pd.to_numeric(df["amount_paid"], errors="coerce")
        - pd.to_numeric(df["amount_received"], errors="coerce")
    ).abs()

    df["amount_ratio"] = (
        pd.to_numeric(df["amount_paid"], errors="coerce")
        / pd.to_numeric(df["amount_received"], errors="coerce").replace(0, pd.NA)
    )

    df["amount_paid"] = pd.to_numeric(df["amount_paid"], errors="coerce").fillna(0)
    df["amount_received"] = pd.to_numeric(df["amount_received"], errors="coerce").fillna(0)

    return df


def merge_typology_labels(transactions_df, patterns_df):
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

    merged_df = transactions_df.merge(
        patterns_df[merge_keys + ["typology", "typology_descriptor"]],
        on=merge_keys,
        how="left"
    )

    merged_df["typology"] = merged_df["typology"].fillna("NORMAL_OR_UNLABELED")
    merged_df["typology_descriptor"] = merged_df["typology_descriptor"].fillna("")

    return merged_df


def route_transaction(row):
    # If an IBM typology was successfully merged, use it.
    if row["typology"] != "NORMAL_OR_UNLABELED":
        return row["typology"]

    # Otherwise use rule-based bank transaction routing.
    if row["payment_format"] == "Reinvestment":
        return "SELF_REINVESTMENT"

    if row["same_account"] == 1:
        return "SELF_TRANSFER"

    if row["currency_changed"] == 1:
        return "CURRENCY_EXCHANGE"

    if row["same_bank"] == 0 and row["amount_paid"] >= 10000:
        return "LARGE_CROSS_BANK_TRANSFER"

    if row["same_bank"] == 0:
        return "CROSS_BANK_TRANSFER"

    if row["payment_format"] == "ACH":
        return "ACH_TRANSFER"

    if row["payment_format"] == "Wire":
        return "WIRE_TRANSFER"

    return "OTHER"


def build_router_dataset(
    transactions_csv=TRANSACTIONS_CSV,
    patterns_csv=PATTERNS_CSV,
    output_csv=OUTPUT_CSV
):
    transactions_df, patterns_df = load_files(transactions_csv, patterns_csv)

    transactions_df = add_router_features(transactions_df)

    
    routed_df = merge_typology_labels(transactions_df, patterns_df)

    routed_df["router_type"] = routed_df.apply(route_transaction, axis=1)

    routed_df.to_csv(output_csv, index=False)

    return routed_df


if __name__ == "__main__":
    routed_df = build_router_dataset()
    print(routed_df.head())

