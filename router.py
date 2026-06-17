import pandas as pd

user = '/Users/erichuang/Documents/dev/Python/researchProj'
SMtransactions = pd.read_csv(user + "/LI-Small_Trans.csv")
#HItransactions = pd.read_csv(user)
#SMpatterns = pd.read_csv(user + "LI-Medium_Patterns.txt")
HIpatterns = pd.read_csv(user + "/HI-Medium_Patterns.txt")

with open("HI-Medium_Patterns.txt", "r", encoding="utf-8") as f: #some standardization for shitty commas
    for _ in range(10):
        print(f.readline())

def parseFilePatterns(HIpatterns):
    rows = []
    currentTypology = None
    currentDescriptor = None 

    with open(HIpatterns): 
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                header = line.replace("BEGIN LAUNDERING ATTEMPT - ", "")
            
                if ":" in header:
                    currentTypology, currentDescription = header.split(":", 1)
                    currentTypology = currentTypology.strip()
                    currentDescription = currentDescription.strip()
                else:
                    currentTypology = header.strip()
                    currentDescription = ""

                continue
     
            if line.startswith("END LAUNDERING ATTEMPT"):
                currentTypology = None
                currentDescription = None
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
                "typology_description": currentDescription
                })

            return pd.DataFrame(rows)

patterns_df = parseFilePatterns("HI-Medium_Patterns.txt")

print(patterns_df.head())
print(patterns_df["typology"].value_counts())

