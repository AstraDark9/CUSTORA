import os

import pandas as pd


def create_features(data, reference_date=None):
    """
    Create customer behavioral features for CuStorA.

    `reference_date` is the "as-of" date that Recency and Tenure are
    measured against. It defaults to today (see below) rather than
    silently using the latest LastPurchaseDate in the dataset.

    Why this matters (see project review notes, issue #10): using
    max(LastPurchaseDate) as the reference date means the exact same
    customer gets a different Recency/Tenure depending on what else
    happens to be in the file with them -- upload the same customer
    alongside a dataset whose most recent purchase was in July 2026
    vs. one where it was in December 2026, and their Recency changes
    even though nothing about that customer did. A real retention
    system needs a stable, explicit "as of" date so numbers computed
    today and numbers computed next week (or a filtered subset vs.
    the full file) mean the same thing and are directly comparable.
    """
    data = data.copy()

    # Make sure date columns are datetime
    data["RegistrationDate"] = pd.to_datetime(data["RegistrationDate"], errors="coerce")
    data["LastPurchaseDate"] = pd.to_datetime(data["LastPurchaseDate"], errors="coerce")

    # Reference date
    # Default: today. Can be overridden by an explicit argument (e.g.
    # a future "analyze as of a past date" feature) or the
    # CUSTORA_ANALYSIS_DATE environment variable (useful for
    # reproducible tests/demos that shouldn't depend on the day they
    # happen to run).

    if reference_date is None:
        env_override = os.environ.get("CUSTORA_ANALYSIS_DATE")
        reference_date = (
            pd.to_datetime(env_override) if env_override else pd.Timestamp.today().normalize()
        )
    else:
        reference_date = pd.to_datetime(reference_date)
    print(f'Analysis ("as of") date: {reference_date.date()}')

    # RFM FEATURES

    # Recency
    data["Recency"] = (reference_date - data["LastPurchaseDate"]).dt.days

    # Frequency
    data["Frequency"] = data["TotalPurchases"]

    # Monetary
    data["Monetary"] = data["TotalSpent"]

    # CUSTOMER TENURE

    data["TenureDays"] = (reference_date - data["RegistrationDate"]).dt.days

    # Convert to years
    data["TenureYears"] = (data["TenureDays"] / 365.25).round(2)

    # COMPLAINT RATE

    data["ComplaintRate"] = (data["Complaints"] / data["TotalPurchases"].replace(0, 1)).round(4)

    # RETURN RATE

    data["ReturnRate"] = (data["Returns"] / data["TotalPurchases"].replace(0, 1)).round(4)

    # SPENDING PER PURCHASE

    data["CalculatedAOV"] = (data["TotalSpent"] / data["TotalPurchases"].replace(0, 1)).round(2)

    # ENGAGEMENT CATEGORY

    data["EngagementLevel"] = pd.cut(
        data["EngagementScore"], bins=[-1, 33, 66, 100], labels=["Low", "Medium", "High"]
    )

    # RECENCY CATEGORY

    data["RecencyLevel"] = pd.cut(
        data["Recency"],
        bins=[-1, 30, 90, float("inf")],
        labels=["Recent", "Moderate", "Inactive"],
    )
    return data


def save_features(data):
    output_file = "data/processed/customers_features.csv"
    data.to_csv(output_file, index=False)
    print(f"\nFeature dataset saved to:" f"\n{output_file}")


def run(data=None, save=True, reference_date=None):
    """
    Create behavioral features and (by default) still write the same
    customers_features.csv checkpoint as before. `data=None` reads
    the cleaned dataset from disk, same as before -- pipeline.py
    instead passes the DataFrame cleaning.py already produced in
    memory, so this stage doesn't re-read what the previous stage
    just wrote.
    """
    if data is None:
        data = pd.read_csv("data/processed/customers_cleaned.csv")
    feature_data = create_features(data, reference_date=reference_date)
    if save:
        save_features(feature_data)
    return feature_data


def main():
    print("Loading cleaned dataset...")
    run()
    print("Feature engineering completed.")
    print("\nNew features created:")
    new_features = [
        "Recency",
        "Frequency",
        "Monetary",
        "TenureDays",
        "TenureYears",
        "ComplaintRate",
        "ReturnRate",
        "CalculatedAOV",
        "EngagementLevel",
        "RecencyLevel",
    ]
    for feature in new_features:
        print(f"  - {feature}")


if __name__ == "__main__":
    main()
