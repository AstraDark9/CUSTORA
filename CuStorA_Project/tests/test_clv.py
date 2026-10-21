from src.business.clv_metrics import (
    calculate_clv,
    classify_customer_value,
    determine_business_priority,
)


def test_clv_is_never_negative():
    row = {
        "AverageOrderValue": -50,
        "Frequency": -2,
        "TenureYears": -1,
        "PredictedChurnProbability": 1.5,
    }
    assert calculate_clv(row) >= 0


def test_higher_churn_probability_reduces_estimated_clv():
    base_row = {
        "AverageOrderValue": 100,
        "Frequency": 10,
        "TenureYears": 2,
    }
    low_churn = calculate_clv({**base_row, "PredictedChurnProbability": 0.1})
    high_churn = calculate_clv({**base_row, "PredictedChurnProbability": 0.9})
    assert low_churn > high_churn


def test_classify_customer_value_boundaries():
    assert classify_customer_value(1000, low_threshold=100, high_threshold=500) == "High Value"
    assert classify_customer_value(300, low_threshold=100, high_threshold=500) == "Medium Value"
    assert classify_customer_value(50, low_threshold=100, high_threshold=500) == "Low Value"


def test_high_value_high_churn_customer_is_critical_business_priority():
    row = {"CustomerValue": "High Value", "ChurnRisk": "High", "CustomerState": "Critical Retention"}
    assert determine_business_priority(row) == "Critical Business Priority"


def test_low_value_low_churn_customer_is_normal_business_priority():
    row = {"CustomerValue": "Low Value", "ChurnRisk": "Low", "CustomerState": "Stable"}
    assert determine_business_priority(row) == "Normal Business Priority"
