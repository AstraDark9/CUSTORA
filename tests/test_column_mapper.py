import pandas as pd

from src.data.column_mapper import suggest_mapping, apply_mapping, build_coverage_report


def test_exact_match_after_normalization():
    result = suggest_mapping(["Customer ID", "gender", "TotalSpent"])
    assert result["mapped"]["CustomerID"] == "Customer ID"
    assert result["mapped"]["Gender"] == "gender"
    assert result["mapped"]["TotalSpent"] == "TotalSpent"


def test_synonym_match():
    result = suggest_mapping(["customer_id", "orders", "TotalCharges"])
    assert result["mapped"]["CustomerID"] == "customer_id"
    assert result["mapped"]["TotalPurchases"] == "orders"
    assert result["mapped"]["TotalSpent"] == "TotalCharges"


def test_unrelated_columns_are_not_force_mapped():
    result = suggest_mapping(["tenure", "InternetService", "Contract"])
    # "tenure" (months of subscription) is not the same concept as
    # TotalPurchases (a purchase count), and must not be force-mapped.
    assert "TotalPurchases" not in result["mapped"]
    assert "CustomerID" not in result["mapped"]
    assert "tenure" in result["unmapped_source"]


def test_missing_required_fields_are_reported():
    result = suggest_mapping(["CustomerID"])
    assert "Age" in result["unmapped_canonical"]
    assert "Churn" in result["unmapped_canonical"]


def test_apply_mapping_only_fills_mapped_columns():
    source = pd.DataFrame({"customer_id": [1, 2], "gender": ["Male", "Female"]})
    result = suggest_mapping(list(source.columns))
    mapped_df = apply_mapping(source, result)
    assert "CustomerID" in mapped_df.columns
    assert "Gender" in mapped_df.columns
    assert "Age" not in mapped_df.columns  # not silently filled in


def test_coverage_report_shape():
    result = suggest_mapping(["CustomerID", "TotalSpent"])
    report = build_coverage_report(result, dataset_name="test")
    assert set(report["Status"]) <= {"Mapped", "Unavailable"}
    assert len(report) == 13  # one row per canonical field
