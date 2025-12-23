import os
import joblib
from sklearn.model_selection import train_test_split
from src.data import get_loan_data
from src.model import build_pipeline

def train_and_save():
    # 1. Load Data
    print("--- Loading Banking Data ---")
    df = get_loan_data()
    X = df[['income', 'loan_amount', 'bureau_data']]
    y = df['default']
    
    # 2. Train Model
    print("--- Training Ensemble Pipeline ---")
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    
    # 3. Save Model
    # Create the 'models' folder if it doesn't exist
    os.makedirs('models', exist_ok=True)
    save_path = 'models/loan_pipeline.pkl'
    
    joblib.dump(pipeline, save_path)
    print(f"✅ Model saved successfully at: {save_path}")

if __name__ == "__main__":
    train_and_save()