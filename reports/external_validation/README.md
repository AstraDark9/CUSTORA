# CuStorA External Validation: IBM Telco Customer Churn

**Dataset source:** https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv (IBM/Kaggle "Telco Customer Churn" sample dataset, 7,043 real customer records)

**Customers:** 7043

**Available columns (21):** customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn

This dataset was used exactly as downloaded -- no columns were renamed, reshaped, or invented to make it look like CuStorA's synthetic schema before evaluation (see project review, "Run CuStorA on real data").

## 1. Column mapping

**4 of 13** CuStorA canonical fields honestly map onto this dataset's columns (see `column_mapping_coverage.csv` for the full table):

| CuStorA Field     | Meaning                                           | Status      | Source Column   |
|:------------------|:--------------------------------------------------|:------------|:----------------|
| CustomerID        | Unique identifier for the customer                | Mapped      | customerID      |
| Age               | Customer's age in years                           | Unavailable |                 |
| Gender            | Customer's gender/category label                  | Mapped      | gender          |
| RegistrationDate  | Date the customer relationship started            | Unavailable |                 |
| LastPurchaseDate  | Date of the customer's most recent purchase/order | Unavailable |                 |
| TotalPurchases    | Count of discrete purchase/order events, all-time | Unavailable |                 |
| TotalSpent        | Total revenue from the customer, all-time         | Mapped      | TotalCharges    |
| AverageOrderValue | Average revenue per discrete purchase/order       | Unavailable |                 |
| EngagementScore   | A 0-100 behavioural engagement score              | Unavailable |                 |
| Complaints        | Count of service complaints, all-time             | Unavailable |                 |
| Returns           | Count of product/order returns, all-time          | Unavailable |                 |
| DiscountUsage     | A 0-100 measure of discount/promo reliance        | Unavailable |                 |
| Churn             | Whether the customer has churned (1) or not (0)   | Mapped      | Churn           |

The columns that don't map aren't a mapping-logic gap -- Telco is a subscription business (customers pay a recurring charge for as long as they stay), not a discrete-purchase business (customers place individual orders). Concretely:
- No **TotalPurchases** / **AverageOrderValue**: there's no concept of discrete "orders" here, only a monthly service charge.
- No **RegistrationDate** / **LastPurchaseDate**: Telco has `tenure` (months as a customer) instead of explicit dates -- a genuinely different (and in some ways more useful) way of expressing the same underlying idea, but not a column CuStorA's date-arithmetic feature engineering can consume without changes.
- No **EngagementScore**, **Complaints**, **Returns**, **DiscountUsage**, **Age**: this business simply doesn't track those things; CuStorA does not invent values for them (see project review, "Do not invent missing business data").


## 2. What breaks

Pushing only the 4 honestly-mapped columns through CuStorA's actual pipeline stages, unmodified:

| Stage                                          | Result                                                                |
|:-----------------------------------------------|:----------------------------------------------------------------------|
| src/data/cleaning.py: clean_dataset()          | FAILED: KeyError: 'Age'                                               |
| src/features/engineering.py: create_features() | FAILED: KeyError: 'RegistrationDate'                                  |
| src/rfm/rfm.py: calculate_rfm_scores()         | FAILED: KeyError: 'Recency'                                           |
| src/clustering/kmeans.py: prepare_data()       | FAILED: ValueError: None of CLUSTER_FEATURES exist in the mapped data |

This confirms the project review's core claim directly: CuStorA in its current form is a CuStorA-schema analysis application, not a general customer-data analysis application. Cleaning and feature engineering fail immediately on a real dataset missing the exact columns CuStorA hardcodes (`data['Age']`, `data['RegistrationDate']`, etc.) rather than degrading gracefully -- confirming project review item #6 (generic upload / column mapping) as the largest remaining product gap, now demonstrated on a real dataset rather than asserted from reading the code.


## 3. What still works: the churn methodology, on Telco's own features

CuStorA's churn-modeling *methodology* (train/validation/test split, calibration, validation-selected threshold, one-time final test evaluation -- project review item #1) does not hardcode CuStorA's column names. It was run, completely unmodified, directly on Telco's own native columns (`tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`, and the rest) with the real `Churn` label mapped from Yes/No to 1/0. Full output is under `reports/external_validation/churn_reports/`; headline numbers:

- **Best model:** Logistic Regression

- **Final, one-time test evaluation** (1409 held-out customers): Accuracy=0.781, Precision=0.581, Recall=0.620, F1=0.600, ROC-AUC=0.832, Brier=0.142


Unlike the synthetic dataset's churn labels (generated by rules over the same fields the model trains on), this Churn label reflects real customer behavior CuStorA had no hand in generating -- so, unlike the synthetic-data numbers, an AUC meaningfully above 0.5 here is real evidence the methodology can find genuine signal, not an artifact of the labels being derived from the features. It also runs on 7,043 real customers rather than 1,000 synthetic ones, so the validation-set threshold-selection instability flagged in the leakage-fix README section is much less pronounced here (see `churn_reports/final_test_evaluation.csv` and `churn_reports/decision_thresholds.json`).


## 4. Segmentation on Telco's native numeric features

K-Means on `tenure`, `MonthlyCharges`, `TotalCharges` (the numeric columns with a genuine equivalent in the clustering investigation's "Recency/Frequency/Monetary"-shaped feature set):

|   K |   Silhouette |
|----:|-------------:|
|   2 |       0.4797 |
|   3 |       0.4514 |
|   4 |       0.472  |

### K=2 cluster profiles (raw feature means)

|   Cluster |   tenure |   MonthlyCharges |   TotalCharges |
|----------:|---------:|-----------------:|---------------:|
|         0 |    20.03 |            52.17 |         869.31 |
|         1 |    56.87 |            89.76 |        5084.99 |

As with the synthetic dataset's clustering investigation, a small number of clusters is silhouette-best here too -- on a dataset 7x larger, from a completely different business. That's independent evidence for the same conclusion reached in `reports/clustering_investigation/README.md`: a small number of clusters isn't a CuStorA limitation, it's what this kind of behavioral feature set tends to actually support.


## What this run required adapting

- A rename (`customerID` -> `CustomerID`) and a label re-encoding (`Churn` Yes/No -> 1/0) were needed before `churn_models.run()` would treat the identifier and target columns correctly -- both are mechanical renames of columns that already meant the right thing, not invented data.
- `TotalCharges` needed an explicit `pd.to_numeric(..., errors='coerce')`: it ships as a string column in the source CSV (11 rows are blank strings for brand-new customers with $0 total charges), which would otherwise be treated as a high-cardinality categorical feature instead of a number.
- Nothing else was changed. CuStorA's cleaning and feature-engineering stages were not run against this dataset at all, for the reasons in section 2 above -- this is the honest result, not a limitation of this validation script.
