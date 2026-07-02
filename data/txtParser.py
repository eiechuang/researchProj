import pandas as pd

user = r"C:\Users\016134703\Documents\researchProj\data"
HIpatterns = pd.read_csv(user + "/HI-Small_Patterns.txt")

def parseFilePatterns(HIpatterns):
    rows = []
    currentTypology = None
    currentDescriptor = None 

    with open(HIpatterns, "r", encoding="utf-8") as f: 
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                header = line.replace("BEGIN LAUNDERING ATTEMPT - ", "")
            
                if ":" in header:
                    currentTypology, currentDescriptor = header.split(":", 1)
                    currentTypology = currentTypology.strip()
                    currentDescriptor = currentDescriptor.strip()
                else:
                    currentTypology = header.strip()
                    currentDescriptor = ""

                continue
     
            if line.startswith("END LAUNDERING ATTEMPT"):
                currentTypology = None
                currentDescriptor = None
                continue
        
            parts = line.split(",")

            if len(parts) != 11:
                print(f"Skipping malformed line: {line}")

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
                "typology": currentTypology,
                "typology_descriptor": currentDescriptor
                })

    return pd.DataFrame(rows)

patterns_df = parseFilePatterns("HI-Small_Patterns.txt")

print(patterns_df.head())
print(patterns_df["typology"].value_counts())
print (patterns_df.shape)

patterns_df.to_csv("routed_patterns.csv", index=False)