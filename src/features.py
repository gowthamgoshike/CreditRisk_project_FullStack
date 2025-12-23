import json
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Constant to prevent division by zero
EPSILON = 1e-5 

class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Selects specific columns from the DataFrame.
    """
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.cols]

class JsonParser(BaseEstimator, TransformerMixin):
    """
    Parses JSON strings into numerical columns.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]

        parsed_list = []
        for json_str in X:
            try:
                data = json.loads(str(json_str))
                parsed_list.append([
                    data.get('credit_score', 0),
                    data.get('late_payments', 0)
                ])
            except (json.JSONDecodeError, TypeError):
                parsed_list.append([0, 0])
        
        return pd.DataFrame(parsed_list, columns=['credit_score', 'late_payments'])

class IncomeToLoanRatio(BaseEstimator, TransformerMixin):
    """
    Calculates Debt-to-Income Ratio.
    Handles unit mismatches (Thousands vs Raw Dollars).
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        
        # Heuristic: If median loan is small (<1000), it's in 'Thousands' unit.
        if X['loan_amount'].median() < 1000:
            loan_in_dollars = X['loan_amount'] * 1000
        else:
            loan_in_dollars = X['loan_amount']

        ratio = loan_in_dollars / (X['income'] + EPSILON)
        
        return pd.DataFrame(ratio, columns=['income_loan_ratio'])