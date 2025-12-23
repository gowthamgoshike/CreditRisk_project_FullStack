# 📡 RiskRadar AI: Loan Default Prediction System

RiskRadar is an **End-to-End Machine Learning Application** designed to predict loan default risks. 

Unlike standard "toy projects," this system simulates a real-world fintech environment by handling **complex data structures (JSON)**, implementing **custom feature engineering pipelines**, and deploying the model via a **Streamlit frontend**.

## 🚀 Key Features

* **Hybrid Data Engineering:** Custom logic to parse nested JSON strings (simulating raw API data from credit bureaus).
* **Production-Ready Pipelines:** Modular `scikit-learn` pipelines using Custom Transformers (`JsonParser`, `IncomeToLoanRatio`).
* **Ensemble Learning:** Uses a **Voting Classifier** (Logistic Regression + Random Forest) for robust predictions.
* **Full-Stack Interface:** Interactive web dashboard built with **Streamlit** for real-time risk analysis.



## 🛠️ Installation & Setup

1.  **Install Dependencies**
    pip install -r requirements.txt

2.  **Train the Model**
    This script downloads real-world data, trains the ensemble model, and saves it as `models/loan_pipeline.pkl`.
    python train_model.py

3.  **Run the Application**
    Launch the web interface locally.
    streamlit run app.py

## 🧠 Model Architecture

The system uses a **Feature Union** architecture to process mixed data types simultaneously:

1.  **JSON Branch:** Extracts `Credit Score` and `Late Payments` from text strings.
2.  **Ratio Branch:** Calculates `Debt-to-Income Ratio` while handling unit mismatches (Thousands vs. Raw Dollars).
3.  **Ensemble:** A Soft Voting Classifier averages the probabilities of:
    * *Logistic Regression* (Linear relationships)
    * *Random Forest* (Non-linear complex patterns)

## 📊 Dataset

This project uses the **Loan Prediction Dataset** (sourced from GitHub).
* **Inputs:** Income, Loan Amount, Credit History (Simulated as JSON).
* **Target:** Loan Status (Approved/Rejected).

