from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

# IMPORT ALL FROM FEATURES.PY
from src.features import JsonParser, IncomeToLoanRatio, ColumnSelector

def build_pipeline():
    # 1. JSON Branch
    json_branch = Pipeline([
        ('selector', ColumnSelector('bureau_data')),
        ('parser', JsonParser()),
        ('scaler', StandardScaler())
    ])

    # 2. Ratio Branch
    ratio_branch = Pipeline([
        ('selector', ColumnSelector(['income', 'loan_amount'])),
        ('ratio_gen', IncomeToLoanRatio()),
        ('scaler', StandardScaler())
    ])

    # 3. Combine
    preprocessor = FeatureUnion([
        ('json_features', json_branch),
        ('ratio_features', ratio_branch)
    ])

    # 4. Ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ],
        voting='soft'
    )

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', ensemble)
    ])
    
    return full_pipeline