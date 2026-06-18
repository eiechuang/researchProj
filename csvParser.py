import pandas as pd

user = "/Users/erichuang/Documents/dev/Python/researchproj2"
file = "/LI-Small_Trans.csv"
input_path = user + file


def fileClean(input_csv):
    df = pd.read_csv(input_csv)

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
    df["amount_received"] = pd.to_numeric(df["amount_received"], errors="coerce")
    df["amount_paid"] = pd.to_numeric(df["amount_paid"], errors="coerce")
    df["is_laundering"] = pd.to_numeric(df["is_laundering"], errors="coerce").fillna(0).astype(int)

    return df


def newFeatures(df): 
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
    )

    df["is_reinvestment"] = (df["payment_format"] == "Reinvestment").astype(int)
    df["is_ach"] = (df["payment_format"] == "ACH").astype(int)
    df["is_wire"] = (df["payment_format"] == "Wire").astype(int)
    df["is_cheque"] = (df["payment_format"] == "Cheque").astype(int)
    df["is_credit_card"] = (df["payment_format"] == "Credit Card").astype(int)

    return df


def route_transaction(row):
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


def build_csv_router(input_csv, output_csv="routed_transactions.csv"):
    df = fileClean(input_csv)
    df = newFeatures(df)

    df["router_type"] = df.apply(route_transaction, axis=1)

    df.to_csv(output_csv, index=False)

    print("Saved routed CSV to:", output_csv)
    print()
    print("Router type counts:")
    print(df["router_type"].value_counts())
    print()
    print("Laundering counts by router type:")
    print(pd.crosstab(df["router_type"], df["is_laundering"]))

    return df


if __name__ == "__main__":
    routed_df = build_csv_router(
        input_csv=input_path,
        output_csv="routed_transactions.csv"
    )

    print(routed_df.head())