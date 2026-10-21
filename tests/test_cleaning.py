import pandas as pd

from src.data.cleaning import clean_dataset


def _base_row(**overrides):
    row = {
        "CustomerID": "CUST1",
        "Age": 30,
        "Gender": "Male",
        "RegistrationDate": "2022-01-01",
        "LastPurchaseDate": "2023-01-01",
        "TotalPurchases": 5,
        "TotalSpent": 500,
        "AverageOrderValue": 100,
        "EngagementScore": 60,
        "Complaints": 0,
        "Returns": 0,
        "DiscountUsage": 1,
        "Churn": 0,
    }
    row.update(overrides)
    return row


def test_cleaning_removes_duplicate_customer_ids():
    data = pd.DataFrame([_base_row(CustomerID="CUST1"), _base_row(CustomerID="CUST1")])
    cleaned = clean_dataset(data)
    assert len(cleaned) == 1


def test_cleaning_treats_negative_values_as_missing_then_fills_them():
    data = pd.DataFrame(
        [
            _base_row(CustomerID="CUST1", Complaints=-5),
            _base_row(CustomerID="CUST2", Complaints=2),
        ]
    )
    cleaned = clean_dataset(data)
    # Negative values are converted to missing then filled with the
    # median, so nothing negative should survive.
    assert (cleaned["Complaints"] >= 0).all()


def test_cleaning_rejects_unrealistic_ages():
    data = pd.DataFrame(
        [
            _base_row(CustomerID="CUST1", Age=5),
            _base_row(CustomerID="CUST2", Age=30),
            _base_row(CustomerID="CUST3", Age=150),
        ]
    )
    cleaned = clean_dataset(data)
    # Ages outside 18-100 get replaced with the column median (30),
    # so every remaining age should be within the valid adult range.
    assert cleaned["Age"].between(18, 100).all()


def test_cleaning_keeps_any_nonblank_gender_as_a_generic_category():
    # Gender is treated generically now -- any non-blank value is kept
    # (trimmed/title-cased), not forced into a fixed Male/Female list.
    data = pd.DataFrame(
        [
            _base_row(CustomerID="CUST1", Gender="Male"),
            _base_row(CustomerID="CUST2", Gender="female"),
            _base_row(CustomerID="CUST3", Gender="non-binary"),
        ]
    )
    cleaned = clean_dataset(data)
    assert set(cleaned["Gender"].unique()) == {"Male", "Female", "Non-Binary"}


def test_cleaning_fills_blank_gender_with_unknown_not_a_guess():
    data = pd.DataFrame(
        [
            _base_row(CustomerID="CUST1", Gender="Male"),
            _base_row(CustomerID="CUST2", Gender="Male"),
            _base_row(CustomerID="CUST3", Gender=""),
        ]
    )
    cleaned = clean_dataset(data)
    assert cleaned.loc[cleaned["CustomerID"] == "CUST3", "Gender"].iloc[0] == "Unknown"


def test_cleaning_does_not_invent_missing_churn_labels():
    # A blank Churn cell must stay missing, not get filled with the
    # column mode -- fabricating a label the business never observed.
    data = pd.DataFrame(
        [
            _base_row(CustomerID="CUST1", Churn=1),
            _base_row(CustomerID="CUST2", Churn=1),
            _base_row(CustomerID="CUST3", Churn=None),
        ]
    )
    cleaned = clean_dataset(data)
    row = cleaned.loc[cleaned["CustomerID"] == "CUST3"]
    assert row["Churn"].isna().iloc[0]
    assert cleaned["Churn"].isna().sum() == 1


def test_cleaning_recomputes_average_order_value_from_totals():
    data = pd.DataFrame(
        [_base_row(CustomerID="CUST1", TotalSpent=1000, TotalPurchases=4, AverageOrderValue=1)]
    )
    cleaned = clean_dataset(data)
    assert cleaned.loc[0, "AverageOrderValue"] == 250.0


def test_cleaning_churn_is_always_binary():
    data = pd.DataFrame(
        [
            _base_row(CustomerID="CUST1", Churn=1),
            _base_row(CustomerID="CUST2", Churn=0),
        ]
    )
    cleaned = clean_dataset(data)
    assert set(cleaned["Churn"].unique()).issubset({0, 1})
