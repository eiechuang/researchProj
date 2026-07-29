import pandas as pd

DATA_FILE = "data/final_ibm_bank_fuzzyrules.csv"
OUTPUT_FILE = "data/feature_distribution_data.csv"

features = [
    "sender_unique_receivers",
    "receiver_unique_senders",
    "sender_daily_tx_count",
    "engineered_risk_score"
]

df = pd.read_csv(DATA_FILE)

needed_cols = ["is_laundering"] + features
df_small = df[needed_cols].copy()

df_small["Class"] = df_small["is_laundering"].map({
    0: "Normal",
    1: "Laundering"
})

df_small = df_small[["Class"] + features]

df_small.to_csv(OUTPUT_FILE, index=False)

print("Saved:", OUTPUT_FILE)
print(df_small.head())