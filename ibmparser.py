import pandas as pd

INPUT_FILE = r"C:\Users\016134703\Documents\researchProj\data/IBMBank.csv"
OUTPUT_FILE = r"C:\Users\016134703\Documents\researchProj\data/ibm_model_ready.csv"


def prepare_ibm_data(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    df = pd.read_csv(input_file)

    model_df = pd.DataFrame()

    model_df["context"] = "BANK_TRANSACTION"

    model_df["amount"] = pd.to_numeric(
        df["amount_paid"],
        errors="coerce"
    ).fillna(0)

    model_df["high_value_flag"] = (
        model_df["amount"] >= 10000
    ).astype(int)

    model_df["very_high_value_flag"] = (
        model_df["amount"] >= 100000
    ).astype(int)

    model_df["same_bank_flag"] = df["same_bank"].fillna(0).astype(int)

    model_df["cross_bank_flag"] = (
        df["same_bank"] == 0
    ).astype(int)

    model_df["same_account_flag"] = df["same_account"].fillna(0).astype(int)

    model_df["currency_changed_flag"] = df["currency_changed"].fillna(0).astype(int)

    model_df["amount_difference"] = pd.to_numeric(
        df["amount_difference"],
        errors="coerce"
    ).fillna(0)

    if "amount_ratio" in df.columns:
        model_df["amount_ratio"] = pd.to_numeric(
            df["amount_ratio"],
            errors="coerce"
        ).fillna(1)
    else:
        model_df["amount_ratio"] = 1

    model_df["payment_format"] = df["payment_format"].fillna("UNKNOWN")

    if "router_type" in df.columns:
        model_df["typology"] = df["router_type"].fillna("BANK_UNKNOWN")
    else:
        model_df["typology"] = "BANK_UNKNOWN"

    # Shared columns that TBML and real estate will also use later
#    model_df["cross_border_flag"] = model_df["currency_changed_flag"]
#    model_df["high_risk_country_flag"] = 0
#    model_df["third_party_flag"] = 0
#    model_df["entity_obfuscation_flag"] = 0
#    model_df["rapid_movement_flag"] = 0
#    model_df["pricing_anomaly_flag"] = 0

 #   model_df["risk_score"] = (
 #       model_df["high_value_flag"] * 1.0
 #       + model_df["very_high_value_flag"] * 1.5
 #       + model_df["cross_bank_flag"] * 0.75
 #       + model_df["currency_changed_flag"] * 1.0
 #       + model_df["same_account_flag"] * 0.25
 #   )


    model_df["is_laundering"] = df["is_laundering"].astype(int)

    model_df.to_csv(output_file, index=False)

    print("Saved:", output_file)
    print("Shape:", model_df.shape)
    print()
    print("Class counts:")
    print(model_df["is_laundering"].value_counts())
    print()
    print("Typology counts:")
    print(model_df["typology"].value_counts().head(20))

    return model_df


if __name__ == "__main__":
    prepare_ibm_data()