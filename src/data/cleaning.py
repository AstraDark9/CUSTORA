"""
Cleans the raw customer dataset before any analysis runs.

Steps: drop duplicate customers, fix bad/negative values, parse
dates, fill missing values, and recompute AverageOrderValue so
everything downstream (RFM, clustering, churn) gets clean input.

Two changes worth calling out explicitly (see project review notes):

1. Missing Churn labels are NOT invented. Earlier versions filled a
   blank Churn cell with the column mode, which quietly manufactures
   a ground-truth label the business never actually observed. That
   silently taught the churn model on fabricated data. Now a missing
   Churn value stays missing; churn_models.py already excludes rows
   with no label from training (see prepare_features there) and can
   still score them for a prediction once a model exists.

2. Gender is treated as a generic category, not restricted to
   Male/Female. Any non-blank value is trimmed/title-cased and kept
   as-is; only a genuinely blank/missing cell is filled, and it's
   filled with "Unknown" rather than silently overwritten with
   whatever value happens to be most common in the file.

Every fix applied is also counted and returned as a "Data Health"
report (see build_data_health_report / clean_dataset) so the amount
of guessing this step does is visible instead of hidden inside a
print statement.
"""

from pathlib import Path

import pandas as pd

INPUT_FILE = "data/raw/customers_messy.csv"
OUTPUT_FILE = "data/processed/customers_cleaned.csv"
DATA_HEALTH_REPORT_FILE = "reports/data_health.csv"

NUMERIC_COLUMNS = [
    "Age",
    "TotalPurchases",
    "TotalSpent",
    "AverageOrderValue",
    "EngagementScore",
    "Complaints",
    "Returns",
    "DiscountUsage",
]
DATE_COLUMNS = ["RegistrationDate", "LastPurchaseDate"]


def clean_dataset(data):
    data = data.copy()
    original_rows = len(data)
    health = {}
    print("\n" + "=" * 50)
    print("             CUSTORA DATA CLEANING")
    print("=" * 50)
    health["rows_before"] = original_rows
    health["missing_values_detected"] = int(data.isnull().sum().sum())

    # 1. Drop duplicate customers, keep the first occurrence
    before = len(data)
    data = data.drop_duplicates(subset="CustomerID", keep="first")
    health["duplicate_customers_removed"] = before - len(data)
    print(f"\nDuplicate customers removed: {health['duplicate_customers_removed']}")

    # 2. Force numeric columns to be numeric (bad values -> NaN)
    non_numeric_coerced = 0
    for column in NUMERIC_COLUMNS + ["Churn"]:
        if column in data.columns:
            before_na = data[column].isna().sum()
            data[column] = pd.to_numeric(data[column], errors="coerce")
            non_numeric_coerced += int(data[column].isna().sum() - before_na)
    health["non_numeric_values_coerced"] = non_numeric_coerced

    # 3. Negative values don't make sense for these columns -> treat as missing
    negative_fixed = 0
    for column in NUMERIC_COLUMNS:
        if column in data.columns:
            negative_mask = data[column] < 0
            negative_fixed += int(negative_mask.sum())
            data.loc[negative_mask, column] = pd.NA
    health["negative_values_converted_to_missing"] = negative_fixed
    print(f"Negative values converted to missing: {negative_fixed}")

    # 4. Age must be a realistic adult age
    unrealistic_age_mask = (data["Age"] < 18) | (data["Age"] > 100)
    health["unrealistic_ages_cleared"] = int(unrealistic_age_mask.sum())
    data.loc[unrealistic_age_mask, "Age"] = pd.NA

    # 5. Gender is a generic category: trim/title-case whatever is
    #    there, don't restrict to a fixed list of values. Only a
    #    truly blank cell counts as "missing".
    data["Gender"] = data["Gender"].astype("string").str.strip().str.title()
    data.loc[data["Gender"].isin(["", "Nan", "None", "N/A", "Na"]), "Gender"] = pd.NA
    health["gender_missing"] = int(data["Gender"].isna().sum())

    # 6. Parse dates (unparseable strings become NaT)
    invalid_dates_corrected = 0
    for column in DATE_COLUMNS:
        before_na = data[column].isna().sum()
        data[column] = pd.to_datetime(data[column], errors="coerce")
        invalid_dates_corrected += int(data[column].isna().sum() - before_na)
    health["invalid_dates_corrected"] = invalid_dates_corrected

    # 7. Fill missing numeric values with the column median
    numeric_values_imputed = 0
    for column in NUMERIC_COLUMNS:
        if data[column].isna().any():
            numeric_values_imputed += int(data[column].isna().sum())
            data[column] = data[column].fillna(data[column].median())
    health["numeric_values_imputed_with_median"] = numeric_values_imputed

    # 8. Fill missing Gender with "Unknown" -- do NOT invent a gender
    #    by copying the most common value in the file.
    if data["Gender"].isna().any():
        data["Gender"] = data["Gender"].fillna("Unknown")

    # 9. A missing date is flagged, not invented.
    #
    # A blank LastPurchaseDate doesn't mean "this customer purchased
    # around the median date" -- it might mean the business simply
    # never recorded one, and for churn analysis that missingness can
    # itself be meaningful (e.g. a customer who never completed a
    # purchase at all). Silently filling it with the median date was
    # asserting a fact CuStorA doesn't actually know (see project
    # review notes, issue #11).
    #
    # Two explicit boolean flag columns record which rows had a
    # missing date, *before* any fill happens, so anything downstream
    # can filter on or explain by them. The median fill still happens
    # after the flag is recorded, because most of CuStorA's numeric
    # feature engineering (Recency, TenureYears, ...) needs *some*
    # date value to compute on -- but it's now a documented,
    # inspectable fallback rather than a hidden one.
    dates_imputed = 0
    for column in DATE_COLUMNS:
        flag_column = f"{column}Imputed"
        data[flag_column] = data[column].isna()
        if data[column].isna().any():
            dates_imputed += int(data[column].isna().sum())
            data[column] = data[column].fillna(data[column].dropna().median())
    health["dates_imputed_with_median"] = dates_imputed

    # 10. Churn must be 0, 1, or genuinely missing. A missing Churn
    #     value is NOT filled with a guess -- it's left as NaN so it
    #     can be excluded from model training later, and the count is
    #     reported so it's visible how many labels are missing.
    health["churn_labels_missing"] = int(data["Churn"].isna().sum())
    valid_churn = data["Churn"].notna()
    data.loc[valid_churn, "Churn"] = data.loc[valid_churn, "Churn"].round().clip(lower=0, upper=1)
    print(
        f"\nChurn labels missing (kept missing, excluded from training): "
        f"{health['churn_labels_missing']}"
    )

    # 11. Recompute AverageOrderValue from the (now clean) totals, instead
    #     of trusting whatever value was originally in the file
    data["AverageOrderValue"] = (data["TotalSpent"] / data["TotalPurchases"].replace(0, 1)).round(2)

    # 12. Sort for a stable, predictable row order
    data = data.sort_values(by="CustomerID").reset_index(drop=True)
    health["rows_after"] = len(data)
    health["rows_removed"] = original_rows - len(data)
    health["remaining_missing_values"] = int(data.isnull().sum().sum())
    print(f"\nOriginal rows : {original_rows}")
    print(f"Final rows    : {len(data)}")
    print(f"Rows removed  : {health['rows_removed']}")
    print(f"Remaining missing values: {health['remaining_missing_values']}")
    print("\nCleaning completed.")
    print("=" * 50)
    data.attrs["data_health"] = health
    return data


def build_data_health_report(health):
    """Turn the health dict collected during cleaning into a small,
    presentable table -- this is what item 15/17 of the project
    review calls "Data Health": a visible account of how much of the
    dataset was guessed at, rather than a silent cleaning step."""
    labels = {
        "rows_before": "Rows before cleaning",
        "rows_after": "Rows after cleaning",
        "rows_removed": "Rows removed",
        "missing_values_detected": "Missing values detected (before cleaning)",
        "duplicate_customers_removed": "Duplicate customers removed",
        "non_numeric_values_coerced": "Non-numeric values coerced to missing",
        "negative_values_converted_to_missing": "Negative values converted to missing",
        "unrealistic_ages_cleared": "Unrealistic ages cleared (outside 18-100)",
        "gender_missing": "Missing gender values found",
        "invalid_dates_corrected": "Invalid dates corrected",
        "numeric_values_imputed_with_median": "Numeric values imputed with median",
        "dates_imputed_with_median": "Dates imputed with median date",
        "churn_labels_missing": "Missing churn labels (excluded from training, not guessed)",
        "remaining_missing_values": "Missing values remaining after cleaning",
    }
    rows = [{"Metric": label, "Value": health.get(key, 0)} for key, label in labels.items()]
    return pd.DataFrame(rows)


def save_data_health_report(health, output_file=DATA_HEALTH_REPORT_FILE):
    report = build_data_health_report(health)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    print(f"\nData Health report saved to:\n{output_file}")
    print("\nData Health")
    print(report.to_string(index=False))
    return report


def save_cleaned_dataset(data, output_file=OUTPUT_FILE):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to:\n{output_file}")


def run(data=None, save=True):
    """
    Clean a customer dataset and (by default) still write the same
    on-disk checkpoint (customers_cleaned.csv) and Data Health report
    as before -- callers that already have the raw DataFrame in memory
    (pipeline.py) just don't have to round-trip it through disk to get
    here, and don't have to re-read what this stage just wrote to hand
    it to the next stage. `data=None` reads INPUT_FILE, same as
    before, so standalone/CLI use (main(), below) is unchanged.
    """
    if data is None:
        data = pd.read_csv(INPUT_FILE)
    cleaned_data = clean_dataset(data)
    if save:
        save_cleaned_dataset(cleaned_data)
        save_data_health_report(cleaned_data.attrs.get("data_health", {}))
    return cleaned_data


def main():
    print("Loading messy dataset...")
    run()


if __name__ == "__main__":
    main()
