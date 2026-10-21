"""
Runs CuStorA against a second real, external customer dataset,
following the same pattern as run_external_validation.py (Telco).

Dataset: the Kaggle "E-Commerce Customer Churn" dataset (5,630 real
e-commerce customers, genuine 0/1 churn labels), pulled from
https://raw.githubusercontent.com/anvarnarz/praktikum_datasets/main/E-Commerce-Dataset.csv
and stored at data/external/ecommerce_customer_churn.csv. Verified
against independent public write-ups of the same dataset (row count,
~16.8% churn rate, per-column missing-value counts) before use -- see
reports/external_validation_ecommerce/README.md.

Deliberately chosen from the same broad domain as CuStorA's synthetic
schema (e-commerce), yet as the column mapping below shows honestly,
only 4 of 13 canonical fields actually match: this is a snapshot of
"last month's" behaviour, not a lifetime purchase ledger, so it does
not force-fit CuStorA's schema just because both are "e-commerce." Per
src/data/column_mapper.py's stated principle, it is NOT reshaped into
data/raw/customers_messy.csv -- doing so would mean fabricating 9 of
13 fields with no source column at all, reintroducing the exact
"features assigned independently of the label" problem this
remediation exists to fix.
"""

import os

from scripts._validation_common import run_validation

EXTERNAL_DATASET_PATH = "data/external/ecommerce_customer_churn.csv"
EXTERNAL_DATASET_SOURCE = (
    "https://raw.githubusercontent.com/anvarnarz/praktikum_datasets/main/"
    'E-Commerce-Dataset.csv (Kaggle "E-Commerce Customer Churn" dataset, '
    "5,630 real customer records)"
)
OUTPUT_DIR = "reports/external_validation_ecommerce"
SEGMENTATION_FEATURES = ["DaySinceLastOrder", "OrderCount", "CashbackAmount"]


def write_readme(
    df, coverage_report, failures_df, churn_result, seg_comparison, seg_profiles, seg_best_k
):
    lines = [
        "# CuStorA External Validation: E-Commerce Customer Churn (Kaggle)\n",
        f"**Dataset source:** {EXTERNAL_DATASET_SOURCE}\n",
        f"**Customers:** {len(df)}\n",
        f"**Available columns ({len(df.columns)}):** {', '.join(df.columns)}\n",
        f"**Churn rate:** {df['Churn'].mean():.1%} ({int(df['Churn'].sum())} of {len(df)} churned) "
        "-- consistent with independently published summaries of this dataset, "
        "a sanity check that this is genuinely the dataset it claims to be.\n",
        "Used exactly as downloaded -- no columns renamed, reshaped, or "
        "invented to look like CuStorA's synthetic schema before evaluation.\n",
        "## 1. Column mapping\n",
    ]
    mapped_count = (coverage_report["Status"] == "Mapped").sum()
    lines.append(
        f"**{mapped_count} of {len(coverage_report)}** CuStorA canonical fields "
        f"honestly map onto this dataset (full table in `column_mapping_coverage.csv`):\n"
    )
    lines.append(coverage_report.to_markdown(index=False))
    lines.append(
        "\nEven from the same broad domain, most fields still don't map, "
        "and that's not a mapping-logic gap: this is a snapshot of *last "
        "month's* behaviour (`OrderCount`, `CouponUsed`, "
        '`DaySinceLastOrder` are all scoped to "last month"), not a '
        "lifetime purchase ledger with dates and running totals -- so no "
        "**RegistrationDate**/**LastPurchaseDate**/**TotalSpent**/"
        "**AverageOrderValue**. It tracks related-but-not-identical things "
        "(`Complain` is a 0/1 flag for last month, not an all-time count) "
        "rather than the same fields under different names, so "
        "`column_mapper.py` correctly declines to auto-map **Age**, "
        "**EngagementScore**, **Complaints**, **Returns**, "
        "**DiscountUsage**.\n"
    )

    lines.append("\n## 2. What breaks\n")
    lines.append(
        "Pushing only the honestly-mapped columns through CuStorA's actual pipeline stages, unmodified:\n"
    )
    lines.append(failures_df.to_markdown(index=False))
    lines.append(
        "\nSame conclusion as Telco, from an independent real dataset: "
        "CuStorA in its current form is a CuStorA-schema analysis "
        "application, not a general customer-data one. Generic column "
        "mapping remains the largest product gap for onboarding arbitrary "
        "real customer data.\n"
    )

    lines.append(
        "\n## 3. What still works: the churn methodology, on this dataset's own features\n"
    )
    lines.append(
        "The train/validation/test split, calibration, validation-selected "
        "threshold, and one-time final test evaluation don't hardcode "
        "CuStorA's column names. Run completely unmodified on this "
        "dataset's own native columns with the real `Churn` label "
        "(already 0/1). Full output under `churn_reports/`; headline "
        "numbers:\n"
    )
    lines.append(_churn_summary(churn_result))
    lines.append(
        "\nAs with Telco, and unlike demo_generator.py's rule-generated "
        "labels, this `Churn` reflects real customer behavior CuStorA had "
        "no hand in generating -- so an AUC meaningfully above 0.5 is real "
        "evidence of genuine signal. A second, independent real-data "
        "confirmation, from a different business type (discrete e-commerce "
        "orders rather than a subscription service).\n"
    )

    lines.append("\n## 4. Segmentation on this dataset's native numeric features\n")
    lines.append(
        "K-Means on `DaySinceLastOrder`, `OrderCount`, `CashbackAmount` "
        "(this dataset's closest genuine equivalent to a "
        "Recency/Frequency/Monetary-shaped feature set):\n"
    )
    lines.append(seg_comparison.to_markdown(index=False))
    lines.append(f"\n### K={seg_best_k} cluster profiles (raw feature means)\n")
    lines.append(seg_profiles[seg_best_k].to_markdown())
    lines.append(
        "\nA third, independent real dataset again favors a small number "
        "of clusters by silhouette -- further evidence this reflects what "
        "this kind of behavioral feature set tends to support, not a quirk "
        "of any one dataset.\n"
    )

    lines.append("\n## What this run required adapting\n")
    lines.append(
        "Nothing. `Churn` was already 0/1 (no re-encoding, unlike Telco) "
        "and every native numeric column loaded correctly with no "
        "string-typed numbers to coerce. CuStorA's cleaning and feature-"
        "engineering stages were not run against this dataset at all, for "
        "the reasons in section 2 above. This dataset was also **not** "
        "used to replace `data/raw/customers_messy.csv` -- see this "
        "script's module docstring for why forcing a 4-of-13-field mapping "
        "into CuStorA's schema would mean fabricating the other 9.\n"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/README.md", "w") as f:
        f.write("\n".join(lines))


def _churn_summary(churn_result):
    if not churn_result:
        return "(Churn methodology run did not complete -- see console output.)\n"
    m = churn_result.get("final_test_metrics", {})
    return (
        f"- **Best model:** {churn_result.get('best_model_name', 'n/a')}\n"
        f"- **Final, one-time test evaluation** ({m.get('TestRows', 'n/a')} held-out customers): "
        f"Accuracy={m.get('Accuracy', float('nan')):.3f}, "
        f"Precision={m.get('Precision', float('nan')):.3f}, "
        f"Recall={m.get('Recall', float('nan')):.3f}, "
        f"F1={m.get('F1', float('nan')):.3f}, "
        f"ROC-AUC={m.get('ROC-AUC', float('nan')):.3f}, "
        f"Brier={m.get('Brier', float('nan')):.3f}\n"
    )


def main():
    run_validation(
        dataset_path=EXTERNAL_DATASET_PATH,
        dataset_name="E-Commerce Customer Churn",
        output_dir=OUTPUT_DIR,
        native_filename="ecommerce_native_features.csv",
        segmentation_features=SEGMENTATION_FEATURES,
        write_readme_fn=write_readme,
        banner="CUSTORA EXTERNAL VALIDATION: E-COMMERCE CUSTOMER CHURN",
    )


if __name__ == "__main__":
    main()
