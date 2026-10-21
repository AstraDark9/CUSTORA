import pandas as pd

from src.features.engineering import create_features


def _clean_frame():
    return pd.DataFrame(
        [
            {
                "CustomerID": "CUST1",
                "RegistrationDate": "2020-01-01",
                "LastPurchaseDate": "2023-01-01",
                "TotalPurchases": 10,
                "TotalSpent": 1000,
                "EngagementScore": 80,
                "Complaints": 2,
                "Returns": 1,
            },
            {
                "CustomerID": "CUST2",
                "RegistrationDate": "2022-06-01",
                "LastPurchaseDate": "2022-12-01",
                "TotalPurchases": 0,
                "TotalSpent": 0,
                "EngagementScore": 10,
                "Complaints": 0,
                "Returns": 0,
            },
        ]
    )


def test_recency_is_zero_when_reference_date_matches_last_purchase():
    # Recency is measured against an explicit "as of" reference date
    # now (see project review notes, issue #10), not silently against
    # max(LastPurchaseDate) in whatever file happens to be loaded. Pass
    # the reference date explicitly here to test that calculation.
    features = create_features(_clean_frame(), reference_date="2023-01-01")
    assert features.loc[features["CustomerID"] == "CUST1", "Recency"].iloc[0] == 0


def test_reference_date_defaults_to_today_not_max_last_purchase():
    # With no reference_date argument, Recency should be measured
    # against today -- not the dataset's own max LastPurchaseDate --
    # so the same customer's Recency doesn't silently change depending
    # on what other rows happen to be in the file with them.
    features = create_features(_clean_frame())
    expected_recency = (pd.Timestamp.today().normalize() - pd.Timestamp("2023-01-01")).days
    assert features.loc[features["CustomerID"] == "CUST1", "Recency"].iloc[0] == expected_recency


def test_complaint_rate_does_not_divide_by_zero():
    features = create_features(_clean_frame())
    # CUST2 has zero purchases; ComplaintRate must not be NaN/inf.
    row = features[features["CustomerID"] == "CUST2"].iloc[0]
    assert row["ComplaintRate"] == 0
    assert pd.notna(row["ComplaintRate"])


def test_engagement_level_buckets_are_valid():
    features = create_features(_clean_frame())
    assert set(features["EngagementLevel"].dropna().unique()).issubset(
        {"Low", "Medium", "High"}
    )
