import pandas as pd
import numpy as np
import json

def get_loan_data():
    """
    Fetches REAL world data from a reliable GitHub source.
    """
    # NEW WORKING URL (The previous one was deleted)
    url = "https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv"
    
    try:
        print(f"Downloading real data from: {url}...")
        df = pd.read_csv(url)
    except Exception as e:
        raise ConnectionError(f"Could not download data. Check internet connection. Error: {e}")

    # 1. Clean Data
    # Drop rows where critical info is missing
    df = df.dropna(subset=['ApplicantIncome', 'LoanAmount', 'Credit_History', 'Loan_Status'])
    
    # 2. Rename columns to match our project pipeline
    df = df.rename(columns={
        'ApplicantIncome': 'income',
        'LoanAmount': 'loan_amount'
    })
    
    # 3. Pack 'Credit_History' into a JSON string to mimic a complex API response
    bureau_data = []
    for history in df['Credit_History']:
        # Map 1.0 (Good History) -> 750 score
        # Map 0.0 (Bad History) -> 400 score
        score = 750 if history == 1.0 else 400
        late = 0 if history == 1.0 else 5
        
        json_entry = json.dumps({
            'credit_score': score,
            'late_payments': late
        })
        bureau_data.append(json_entry)
        
    df['bureau_data'] = bureau_data

    # 4. Process Target (Y=Approved, N=Rejected/Default)
    # We want to predict Default (Risk), so we set N = 1, Y = 0
    df['default'] = df['Loan_Status'].apply(lambda x: 1 if x == 'N' else 0)

    # Return only the columns our Pipeline expects
    return df[['income', 'loan_amount', 'bureau_data', 'default']]