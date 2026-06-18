import pandas as pd 

user = '/Users/erichuang/Documents/dev/Python/researchproj2'
file = '/LI-Small_Trans.csv'
df = pd.readcsv(user+file)

def fileClean (df):  
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

def newFeatures (df): 
    df = df.copy

    df['same_bank'] = (df['from_bank'] == df['to_bank']).astype(int)
    df['same_account'] = (df['from_account'] == df['to_account']).astype(int) 
    df['currency_exchange'] = (df['payment_currency'] != df['payment_currency']).astype(int)
    df['amount_difference'] = (df['amount_paid'] - df['amount_recieved']).astype(int)
    

    df["is_reinvestment"] = (df["payment_format"] == "Reinvestment").astype(int)
    df["is_ach"] = (df["payment_format"] == "ACH").astype(int)
    df["is_wire"] = (df["payment_format"] == "Wire").astype(int)
    df["is_cheque"] = (df["payment_format"] == "Cheque").astype(int)
    df["is_credit_card"] = (df["payment_format"] == "Credit Card").astype(int)

    return df

