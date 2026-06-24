import pandas as pd

DATA_DIR = "/Users/erichuang/Documents/dev/Python/researchproj2/data/"

patterns_csv = "routed_patterns.csv"
transactions_csv = "routed_transactions.csv"
output = "output_transactions.csv"

TRANSACTIONS_CSV = DATA_DIR + transactions_csv
PATTERNS_CSV = DATA_DIR + patterns_csv
OUTPUT_CSV = DATA_DIR + output


def loadFiles(transactions_csv, patterns_csv):
    transactions_df = pd.read_csv(transactions_csv)
    patterns_df = pd.read_csv(patterns_csv)

    return transactions_df, patterns_df

def addRouterFeatures(df): 

    df["same_bank"] = (df["from_bank"] == df["to_bank"]).astype(int)
    df["same_account"] = (df["from_account"] == df["to_account"]).astype(int) 
    df["currency_changed"] = (df["receiving_currency"] != df["payment_currency"]).astype(int)
    df["amount_difference"] = (
        df["amount_paid"] - df["amount_received"]
    ).abs()

    df["amount_ratio"] = (
        df["amount_paid"] / df["amount_received"].replace(0, pd.NA)
    )

    return df

def mergeTypologyLabels(transactions_df, patterns_df):
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

def installNewTypes (row): 
    if row["typology"] != "NORMAL_OR_UNLABELED":
        return row["typology"]
      
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
    transactions_df, patterns_df = loadFiles(
        transactions_csv,
        patterns_csv
    )

    transactions_df = addRouterFeatures(transactions_df)

    routed_df = mergeTypologyLabels(transactions_df, patterns_df)

    routed_df["router_type"] = routed_df.apply(installNewTypes, axis=1)

    routed_df.to_csv(output_csv, index=False)

    print("Saved:", output_csv)
    print()
    print("Router type counts:")
    print(routed_df["router_type"].value_counts())
    print()
    print("Typology counts:")
    print(routed_df["typology"].value_counts())
    print()
    print("Laundering counts by router type:")
    print(pd.crosstab(routed_df["router_type"], routed_df["is_laundering"]))

    return routed_df


if __name__ == "__main__":
    routed_df = build_router_dataset()
    print(routed_df.head())
   




    
