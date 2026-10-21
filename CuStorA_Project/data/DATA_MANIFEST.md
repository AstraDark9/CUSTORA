# Data manifest

What actually backs each dataset under `data/`, and what label it should
be reported under. Every accuracy/metric number produced anywhere in this
repo should be described using the label of whichever file it was computed
on -- see `CuStorA_data_realism_prompt.md` for why the distinction matters.

**Note:** the three `data/external/*` files (real Telco, e-commerce and UCI
Online Retail datasets) ARE included in this package. UCI Online Retail is
the default real-data demo: the app builds customer snapshots from its
transactions and trains/tests a churn model on them (see README, "Two ways
to feed it"). The shipped `data/processed/*`, `models/churn/*` and
`reports/*` come from that real-data run unless stated otherwise in
`reports/run_meta.json`, which records what each run was based on. Telco and
e-commerce are used only for the independent validation reports.

| File | Label | What it is |
|---|---|---|
| `data/raw/customers_messy.csv`, `data/demo/customers_demo.csv` | **synthetic-with-injected-signal** | Produced by `src/data/demo_generator.py`. Not real customer data. Each customer gets a hidden "relationship quality" trait (not in the output columns) that pulls purchases, spend, engagement, complaints, returns, and tenure in a consistent, correlated direction, with independent noise on top of each field; `Churn` is then drawn from a noisy combination of that hidden trait plus the observed fields -- related to them but not perfectly reconstructable, unlike the generator's previous version which derived `Churn` from a literal threshold rule over the same columns fed to the model. `Age` is deliberately independent of everything, including `Churn`. This is a legitimate improvement over pure flat noise (there is real learnable structure in it) but it is still a designed dataset, not observed customer behavior -- metrics from it demonstrate the pipeline's mechanics, not real-world predictive accuracy. See the module docstring in `demo_generator.py` for the full reasoning. |
| `data/external/telco_customer_churn_ibm.csv` | **real** | IBM/Kaggle "Telco Customer Churn" dataset, 7,043 real telecom customers, genuine Yes/No churn labels, downloaded unmodified from `raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d`. Only 4/13 of CuStorA's canonical fields honestly map onto it (subscription business, no discrete orders/dates/complaints/returns). Evaluated on its own native columns, not reshaped into CuStorA's schema; final held-out ROC-AUC 0.83. |
| `data/external/ecommerce_customer_churn.csv` | **real** | Kaggle "E-Commerce Customer Churn" dataset, 5,630 real e-commerce customers, genuine 0/1 churn labels, downloaded unmodified from `raw.githubusercontent.com/anvarnarz/praktikum_datasets`. Row count, churn rate (16.8%), and per-column missing-value counts cross-checked against multiple independent public write-ups of the same dataset before use. Only 4/13 canonical fields honestly map (this is a last-month behavioural snapshot, not a lifetime purchase ledger). Evaluated on its own native columns, not reshaped into CuStorA's schema; final held-out ROC-AUC 0.89. |
| `data/external/online_retail_transactions_uci.csv` | **real** | UCI "Online Retail" dataset (Chen, Sain & Guo, 2012), 530,103 real purchase-line transactions from 4,340 real customers of a UK online gift retailer, Dec 2010-Dec 2011, downloaded unmodified from `raw.githubusercontent.com/eaintkyawthmu/UCI_Online_Retail_Dataset_Cleaned_Version`. Unlike the two snapshot datasets above, this is a genuine per-invoice transaction ledger with real dates, which is what made real temporal features and a proper forward-looking observation-window churn label possible. The derived temporal-snapshot file and its model were removed from this package along with it; held-out ROC-AUC 0.77, PR-AUC 0.53, Lift@10% 2.1-2.2x. |
| `data/processed/*.csv`, `data/raw/customers_previous.csv`, `data/raw/uploads/*` | *(inherits its source's label)* | Pipeline-stage outputs (cleaned / feature-engineered / RFM / clustered / etc.) derived from whichever file was fed into the pipeline for that run. Not a separate data source -- check which raw/demo/external/upload file produced a given processed file before quoting metrics from it. |

## Why no file here is labeled "SDV-synthesized-from-real"

No real seed data (Path 2 of `CuStorA_data_realism_prompt.md`) was
available or needed for this pass -- Path 1 succeeded twice over (two
independent real, public datasets), which the prompt treats as a better
outcome, so Paths 2 and 3 were not exercised in this run.

## Why neither real dataset replaced `customers_messy.csv`

Both real datasets found only honestly map 4 of CuStorA's 13 canonical
fields. Forcing either one into `customers_messy.csv` would mean
fabricating the other 9 fields (Age, RegistrationDate, LastPurchaseDate,
TotalSpent, AverageOrderValue, EngagementScore, Complaints, Returns,
DiscountUsage) -- which is exactly the "don't invent missing business
data" principle `src/data/column_mapper.py` and `src/data/cleaning.py`
already enforce elsewhere in this codebase. Instead, each real dataset is
run through CuStorA's churn-modeling *methodology* on its own native
columns (see *Data and limitations* in README.md), which is enough to confirm
the methodology finds genuine signal on data CuStorA had no hand in
generating -- without pretending a 9-field-fabricated CSV is "real
customer data."
