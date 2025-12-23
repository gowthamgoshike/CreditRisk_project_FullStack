import pandas as pd
import json
from src.features import JsonParser

def test_json_parser_integrity():
    """
    Test that the parser correctly extracts credit score.
    """
    # 1. Create Dummy Input
    fake_data = pd.Series([
        json.dumps({'credit_score': 750, 'late_payments': 0}),
        json.dumps({'credit_score': 500, 'late_payments': 2})
    ])
    
    # 2. Run Transformer
    parser = JsonParser()
    result = parser.transform(fake_data)
    
    # 3. Assertions (The Check)
    assert result.shape == (2, 2)
    assert result.iloc[0]['credit_score'] == 750
    assert result.iloc[1]['late_payments'] == 2

def test_json_parser_error_handling():
    """
    Test that the parser handles broken JSON without crashing.
    """
    bad_data = pd.Series(["THIS IS NOT JSON", ""])
    
    parser = JsonParser()
    result = parser.transform(bad_data)
    
    # Should return zeros (default value), not crash
    assert result.iloc[0]['credit_score'] == 0