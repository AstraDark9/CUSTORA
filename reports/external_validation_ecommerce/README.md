# CuStorA External Validation: E-Commerce Customer Churn (Kaggle)

**Dataset source:** https://raw.githubusercontent.com/anvarnarz/praktikum_datasets/main/E-Commerce-Dataset.csv (Kaggle "E-Commerce Customer Churn" dataset, 5,630 real customer records)

**Customers:** 5630

**Available columns (20):** CustomerID, Churn, Tenure, PreferredLoginDevice, CityTier, WarehouseToHome, PreferredPaymentMode, Gender, HourSpendOnApp, NumberOfDeviceRegistered, PreferedOrderCat, SatisfactionScore, MaritalStatus, NumberOfAddress, Complain, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder, CashbackAmount

**Churn rate:** 16.8% (948 of 5630 churned) -- consistent with independently published summaries of this same dataset, used here as a sanity check that this is genuinely the dataset it claims to be.

This dataset was used exactly as downloaded -- no columns were renamed, reshaped, or invented to make it look like CuStorA's synthetic schema before evaluation.

## 1. Column mapping

**4 of 13** CuStorA canonical fields honestly map onto this dataset's columns (see `column_mapping_coverage.csv` for the full table):

| CuStorA Field     | Meaning                                           | Status      | Source Column   |
|:------------------|:--------------------------------------------------|:------------|:----------------|
| CustomerID        | Unique identifier for the customer                | Mapped      | CustomerID      |
| Age               | Customer's age in years                           | Unavailable |                 |
| Gender            | Customer's gender/category label                  | Mapped      | Gender          |
| RegistrationDate  | Date the customer relationship started            | Unavailable |                 |
| LastPurchaseDate  | Date of the customer's most recent purchase/order | Unavailable |                 |
| TotalPurchases    | Count of discrete purchase/order events, all-time | Mapped      | OrderCount      |
| TotalSpent        | Total revenue from the customer, all-time         | Unavailable |                 |
| AverageOrderValue | Average revenue per discrete purchase/order       | Unavailable |                 |
| EngagementScore   | A 0-100 behavioural engagement score              | Unavailable |                 |
| Complaints        | Count of service complaints, all-time             | Unavailable |                 |
| Returns           | Count of product/order returns, all-time          | Unavailable |                 |
| DiscountUsage     | A 0-100 measure of discount/promo reliance        | Unavailable |                 |
| Churn             | Whether the customer has churned (1) or not (0)   | Mapped      | Churn           |

Even though this dataset is from the same broad domain as CuStorA's synthetic schema (e-commerce), most fields still don't map, and this isn't a mapping-logic gap:
- No **RegistrationDate** / **LastPurchaseDate** / **TotalSpent** / **AverageOrderValue**: this dataset is a snapshot of *last month's* behaviour (`OrderCount`, `CouponUsed`, `DaySinceLastOrder` are all scoped to "last month"), not a lifetime purchase ledger with dates and running totals the way CuStorA's schema expects.
- No **Age**, **EngagementScore**, **Complaints** (count, all-time), **Returns**, **DiscountUsage** (a 0-100 reliance score): this dataset tracks related-but-not-identical things (`Complain` is a 0/1 flag for last month, not an all-time count; `CouponUsed` is a count of coupons, not a 0-100 reliance score) -- genuinely different quantities, not the same field under a different name, so `column_mapper.py` correctly declines to auto-map them (see its module docstring on structural vs. no match).


## 2. What breaks

Pushing only the honestly-mapped columns through CuStorA's actual pipeline stages, unmodified:

| Stage                                          | Result                                                                |
|:-----------------------------------------------|:----------------------------------------------------------------------|
| src/data/cleaning.py: clean_dataset()          | FAILED: KeyError: 'Age'                                               |
| src/features/engineering.py: create_features() | FAILED: KeyError: 'RegistrationDate'                                  |
| src/rfm/rfm.py: calculate_rfm_scores()         | FAILED: KeyError: 'Recency'                                           |
| src/clustering/kmeans.py: prepare_data()       | FAILED: ValueError: None of CLUSTER_FEATURES exist in the mapped data |

Same conclusion as the Telco validation, from an independent real dataset: CuStorA in its current form is a CuStorA-schema analysis application, not a general customer-data analysis application. Generic column mapping (project review item #6) remains the largest product gap for onboarding arbitrary real customer data.


## 3. What still works: the churn methodology, on this dataset's own features

CuStorA's churn-modeling methodology (train/validation/test split, calibration, validation-selected threshold, one-time final test evaluation) does not hardcode CuStorA's column names, and was run completely unmodified directly on this dataset's own native columns (`Tenure`, `HourSpendOnApp`, `OrderCount`, `CouponUsed`, `Complain`, `SatisfactionScore`, `CashbackAmount`, and the rest) with the real `Churn` label already provided as 0/1. Full output is under `reports/external_validation_ecommerce/churn_reports/`; headline numbers:

- **Best model:** Logistic Regression

- **Final, one-time test evaluation** (1126 held-out customers): Accuracy=0.846, Precision=0.530, Recall=0.751, F1=0.621, ROC-AUC=0.893, Brier=0.081


As with Telco, and unlike demo_generator.py's synthetic labels (generated by rules over the same fields the model trains on), this Churn label reflects real customer behavior CuStorA had no hand in generating -- so an AUC meaningfully above 0.5 here is real evidence the methodology finds genuine signal, not an artifact of labels being derived from features. This is a second, independent real-data confirmation alongside the Telco result (see `reports/external_validation/README.md`), from a different business type (discrete e-commerce orders rather than a subscription service).


## 4. Segmentation on this dataset's native numeric features

K-Means on `DaySinceLastOrder`, `OrderCount`, `CashbackAmount` (this dataset's closest genuine equivalent to a Recency/Frequency/Monetary-shaped feature set):

|   K |   Silhouette |
|----:|-------------:|
|   2 |       0.4525 |
|   3 |       0.407  |
|   4 |       0.4287 |

### K=2 cluster profiles (raw feature means)

|   Cluster |   DaySinceLastOrder |   OrderCount |   CashbackAmount |
|----------:|--------------------:|-------------:|-----------------:|
|         0 |                2.96 |         1.77 |           162.24 |
|         1 |                8.31 |         6.03 |           215.8  |

A third, independent real dataset again favors a small number of clusters by silhouette -- further evidence (alongside the synthetic dataset's own investigation and the Telco result) that this reflects what this kind of behavioral feature set tends to support, not a quirk of any one dataset.


## What this run required adapting

- Nothing. `Churn` was already 0/1 (no Yes/No re-encoding needed, unlike Telco), and every native numeric column loaded correctly as numeric with no string-typed number columns to coerce. CuStorA's cleaning and feature-engineering stages were not run against this dataset at all, for the reasons in section 2 above -- this is the honest result, not a limitation of this validation script.
- This dataset was **not** used to replace `data/raw/customers_messy.csv` -- see the module docstring at the top of this script for why forcing a 4-of-13-field mapping into CuStorA's schema would mean fabricating the other 9 fields, which would just reintroduce the label-independent-of-features problem this remediation exists to fix. The value of this dataset is used here, in its own native form (step 3), not as a schema-reshaped replacement input.
