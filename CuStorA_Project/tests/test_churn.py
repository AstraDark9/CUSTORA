import pandas as pd

from src.churn.churn_models import (
    prepare_features,
    build_threshold_table,
    select_best_model,
)


def _segmented_frame():
    return pd.DataFrame(
        {
            "CustomerID": ["CUST1", "CUST2"],
            "RegistrationDate": ["2022-01-01", "2022-01-01"],
            "LastPurchaseDate": ["2023-01-01", "2023-01-01"],
            "Cluster": [0, 1],
            "SegmentName": ["A", "B"],
            "RFM_Segment": ["Champions", "At Risk"],
            "RFM_Score": ["555", "111"],
            "RFM_Total": [15, 3],
            "R_Score": [5, 1],
            "F_Score": [5, 1],
            "M_Score": [5, 1],
            "TotalPurchases": [10, 2],
            "TotalSpent": [1000, 100],
            "TenureDays": [365, 30],
            "TenureYears": [1.0, 0.08],
            "CalculatedAOV": [100, 50],
            "AverageOrderValue": [100, 50],
            "EngagementLevel": ["High", "Low"],
            "RecencyLevel": ["Recent", "Inactive"],
            "Recency": [5, 200],
            "Frequency": [10, 2],
            "Monetary": [1000, 100],
            "EngagementScore": [90, 10],
            "Churn": [0, 1],
        }
    )


def test_churn_features_exclude_the_target_column():
    X, y = prepare_features(_segmented_frame())
    assert "Churn" not in X.columns


def test_churn_features_exclude_segmentation_outputs():
    X, _ = prepare_features(_segmented_frame())
    leaky_columns = {"Cluster", "SegmentName", "RFM_Segment", "RFM_Score"}
    assert not leaky_columns & set(X.columns)


def test_churn_features_exclude_duplicate_signals():
    # TotalPurchases duplicates Frequency, TotalSpent duplicates
    # Monetary -- both should be dropped so the model doesn't see the
    # same signal twice under two names.
    X, _ = prepare_features(_segmented_frame())
    assert "TotalPurchases" not in X.columns
    assert "TotalSpent" not in X.columns
    assert "Frequency" in X.columns
    assert "Monetary" in X.columns


def test_threshold_table_flags_fewer_customers_at_higher_thresholds():
    y_test = pd.Series([1, 1, 1, 0, 0, 0, 0, 0])
    y_probability = pd.Series([0.9, 0.8, 0.6, 0.55, 0.45, 0.3, 0.2, 0.1])
    table = build_threshold_table(y_test, y_probability, thresholds=[0.3, 0.5, 0.7])
    flagged_counts = table.sort_values("Threshold")["CustomersFlagged"].tolist()

    # Raising the threshold should never flag *more* customers.
    assert flagged_counts == sorted(flagged_counts, reverse=True)


def test_select_best_model_orders_by_recall_then_f1():
    results = pd.DataFrame(
        [
            {"Model": "A", "Recall": 0.70, "F1": 0.60, "Accuracy": 0.8, "Precision": 0.5, "ROC-AUC": 0.7},
            {"Model": "B", "Recall": 0.90, "F1": 0.40, "Accuracy": 0.6, "Precision": 0.3, "ROC-AUC": 0.65},
            {"Model": "C", "Recall": 0.90, "F1": 0.55, "Accuracy": 0.7, "Precision": 0.4, "ROC-AUC": 0.68},
        ]
    )
    ranked = select_best_model(results)
    # B and C tie on Recall (0.90); C has the higher F1 so it should win.
    assert ranked.iloc[0]["Model"] == "C"
