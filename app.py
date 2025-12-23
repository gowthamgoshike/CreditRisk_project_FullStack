import streamlit as st
import pandas as pd
import json
import joblib

# IMPORTANT: Must import custom classes so the model knows what they are
from src.features import JsonParser, IncomeToLoanRatio, ColumnSelector

from src.model import ColumnSelector  # <--- It lives here now!
# Page Setup
st.set_page_config(page_title="RiskRadar AI", page_icon="📡", layout="centered")

def load_model():
    """Loads the trained model from the models directory."""
    try:
        return joblib.load('models/loan_pipeline.pkl')
    except FileNotFoundError:
        st.error("❌ Model file not found. Please run 'train_model.py' first.")
        return None

def main():
    st.title("📡 RiskRadar")
    st.subheader("Real-time Loan Default Prediction System")
    st.markdown("---")

    # --- LEFT COLUMN: INPUTS ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.info("👤 Applicant Details")
        # Variable Name: income
        income = st.number_input("Annual Income ($)", 5000, 200000, 50000)
        
        # Variable Name: loan_amt (Fixed!)
        loan_amt = st.number_input("Loan Amount ($)", 1000, 500000, 15000)

    with col2:
        st.info("🏦 Credit Bureau Data")
        # Variable Name: credit_score
        credit_score = st.slider("Credit Score", 300, 850, 720)
        late_payments = st.selectbox("Late Payments (Last 2 yrs)", [0, 1, 2, 3, 4, 5])

    # --- DATA PREPARATION ---
    # 1. Create the JSON string
    bureau_json = json.dumps({
        "credit_score": credit_score,
        "late_payments": late_payments
    })

    # 2. Create the DataFrame (Variables must match inputs above)
    input_df = pd.DataFrame([{
        'income': income,
        'loan_amount': loan_amt,  # <--- This now matches the input variable
        'bureau_data': bureau_json
    }])

    # --- PREDICTION LOGIC ---
    st.markdown("---")
    if st.button("Analyze Risk Profile", use_container_width=True):
        model = load_model()
        
        if model:
            # Get Prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            # Display Logic
            if prediction == 1:
                st.error(f"🛑 **HIGH RISK: REJECT LOAN**")
                st.write(f"Default Probability: **{probability:.1%}**")
            else:
                st.success(f"✅ **LOW RISK: APPROVE LOAN**")
                st.write(f"Safety Score: **{(1-probability):.1%}**")

            # --- EXPLAINABILITY ---
            with st.expander("See Analysis Details"):
                st.write("The model processed the following raw features:")
                st.json({
                    "Income": income,
                    "Loan Requested": loan_amt,
                    "Calculated Ratio": f"{loan_amt/income:.2f}",
                    "Parsed Bureau JSON": bureau_json
                })

if __name__ == "__main__":
    main()