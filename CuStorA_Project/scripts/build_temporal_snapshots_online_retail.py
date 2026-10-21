"""
Builds a temporally-honest churn training set from a real, public
transactional e-commerce dataset: the UCI "Online Retail" dataset
(Daqing Chen, Sai Liang Sain, Kun Guo, Journal of Database Marketing
and Customer Strategy Management, Vol. 19, No. 3, 2012;
https://archive.ics.uci.edu/dataset/352/online+retail), 531,282 real
invoice line items from 4,340 real customers of a UK-based online gift
retailer, Dec 2010 - Dec 2011. Pulled (already-cleaned of cancellations
and missing CustomerIDs by a third party, verified against the known
public row counts for this dataset) from
raw.githubusercontent.com/eaintkyawthmu/UCI_Online_Retail_Dataset_Cleaned_Version
and stored unmodified at data/external/online_retail_transactions_uci.csv.

Unlike the other two external validations in this repo (Telco, and the
Kaggle E-Commerce Customer Churn snapshot -- see DATA_MANIFEST.md),
this dataset has a genuine transaction ledger with per-invoice dates
per customer, spanning 12+ months. That is what makes it possible to
do the thing the synthetic data and the two snapshot-style real
datasets structurally cannot: build actual temporal features (recency
*trends*, not just recency; frequency change; slopes; volatility) and
a real forward-looking observation-window churn label, instead of a
single all-time snapshot. This follows the roadmap's Tier-1
priorities:

  #2  proper churn-label definition: "will this ALREADY-ACTIVE customer
      make zero purchases in the 90 days after cutoff?" -- not "is
      inactive right now".
  #3  temporal recency/frequency/monetary trend features
  #4  behavioral slope features (linear trend of weekly activity)
  #5  volatility features
  #6  customer lifecycle features (tenure, age)
  #11 RFM + RFM trend
  #14 interaction features
  #15 multiple snapshots per customer (not one row per customer)
  #17 leakage prevention: every feature is computed using only
      transactions with InvoiceDate <= cutoff_date; the label is computed
      using only transactions with cutoff_date < InvoiceDate <=
      cutoff_date + 90 days.

Output: data/temporal_snapshots.csv, one row per (customer, cutoff_date)
snapshot, with a `cutoff_date` column so the modeling script can do a
genuine time-based train/validation/test split (roadmap #16) instead of
a random split.
"""
import numpy as np
import pandas as pd

RAW_PATH = "data/external/online_retail_transactions_uci.csv"
OUT_PATH = "data/processed/online_retail_temporal_snapshots.csv"

HORIZON_DAYS = 90          # "will they churn in the next 90 days?"
MIN_HISTORY_DAYS = 60      # need at least this much history before a cutoff is usable
SLOPE_WINDOW_WEEKS = 12    # trailing window used for slope/volatility features

# Monthly cutoffs. Needs >=60 days of prior data (first invoice 2010-12-01)
# and cutoff + 90 days must stay inside the observed data (last invoice
# 2011-12-09), so the latest usable cutoff is ~2011-09-10.
CUTOFFS = [
    "2011-03-31", "2011-04-30", "2011-05-31",
    "2011-06-30", "2011-07-31", "2011-08-31",
]


def load_transactions():
    df = pd.read_csv(RAW_PATH, dtype={"CustomerID": "str"})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    # keep genuine positive-value purchase lines only (return/cancellation
    # rows and free/zero-price giveaway lines are not real purchase signal)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    return df


def weekly_slope_and_volatility(weekly_counts):
    """OLS slope of a weekly count series, and coefficient of variation."""
    n = len(weekly_counts)
    if n < 3 or np.all(weekly_counts == 0):
        return 0.0, 0.0
    x = np.arange(n)
    y = weekly_counts.astype(float)
    x_mean, y_mean = x.mean(), y.mean()
    denom = ((x - x_mean) ** 2).sum()
    slope = 0.0 if denom == 0 else ((x - x_mean) * (y - y_mean)).sum() / denom
    cv = 0.0 if y_mean == 0 else y.std() / y_mean
    return slope, cv


def build_snapshot(df, cutoff_date, horizon_days=HORIZON_DAYS):
    cutoff = pd.Timestamp(cutoff_date)
    label_end = cutoff + pd.Timedelta(days=horizon_days)
    hist = df[df["InvoiceDate"] <= cutoff]
    future = df[(df["InvoiceDate"] > cutoff) & (df["InvoiceDate"] <= label_end)]

    # only "active" customers: at least one purchase in the 90 days
    # BEFORE cutoff. This is the population the churn question is
    # actually about -- a customer who was already gone isn't "at risk
    # of churning", they've already churned; a brand-new customer with
    # no history yet gives the model nothing to learn recency/frequency
    # trends from.
    lookback_start = cutoff - pd.Timedelta(days=MIN_HISTORY_DAYS)
    recently_active_ids = hist[hist["InvoiceDate"] > lookback_start]["CustomerID"].unique()
    hist = hist[hist["CustomerID"].isin(recently_active_ids)]
    rows = []
    future_active_ids = set(future["CustomerID"].unique())
    grouped = hist.groupby("CustomerID")
    for cust_id, g in grouped:
        g = g.sort_values("InvoiceDate")
        first_purchase = g["InvoiceDate"].min()
        last_purchase = g["InvoiceDate"].max()
        invoices = g.drop_duplicates("InvoiceNo")
        recency_days = (cutoff - last_purchase).days
        tenure_days = (cutoff - first_purchase).days
        frequency_total = invoices["InvoiceNo"].nunique()
        monetary_total = g["Revenue"].sum()
        avg_order_value = monetary_total / max(frequency_total, 1)
        unique_products = g["StockCode"].nunique()
        avg_basket_size = g.groupby("InvoiceNo")["Quantity"].sum().mean()
        def window(days_start, days_end):
            # transactions with cutoff-days_start < date <= cutoff-days_end (days_end closer to cutoff, e.g. 0)
            lo = cutoff - pd.Timedelta(days=days_start)
            hi = cutoff - pd.Timedelta(days=days_end)
            w = g[(g["InvoiceDate"] > lo) & (g["InvoiceDate"] <= hi)]
            return w["InvoiceNo"].nunique(), w["Revenue"].sum()
        orders_30d, spend_30d = window(30, 0)
        orders_prev_30d, spend_prev_30d = window(60, 30)
        orders_90d, spend_90d = window(90, 0)
        orders_prev_90d, spend_prev_90d = window(180, 90)
        freq_trend_ratio = orders_30d / (orders_prev_30d + 1)
        spend_trend_ratio = spend_30d / (spend_prev_30d + 1)
        spend_change_30d = spend_30d - spend_prev_30d
        freq_change_90d = orders_90d - orders_prev_90d

        # weekly series over trailing SLOPE_WINDOW_WEEKS for slope/volatility
        window_start = cutoff - pd.Timedelta(weeks=SLOPE_WINDOW_WEEKS)
        wg = g[g["InvoiceDate"] > window_start].copy()
        if len(wg):
            wg["week"] = ((cutoff - wg["InvoiceDate"]).dt.days // 7)
            weekly_orders = (
                wg.drop_duplicates("InvoiceNo")
                .groupby("week")["InvoiceNo"].count()
                .reindex(range(SLOPE_WINDOW_WEEKS), fill_value=0)
                .sort_index(ascending=False)  # oldest week first
                .values
            )
            weekly_spend = (
                wg.groupby("week")["Revenue"].sum()
                .reindex(range(SLOPE_WINDOW_WEEKS), fill_value=0)
                .sort_index(ascending=False)
                .values
            )
        else:
            weekly_orders = np.zeros(SLOPE_WINDOW_WEEKS)
            weekly_spend = np.zeros(SLOPE_WINDOW_WEEKS)
        activity_slope, activity_cv = weekly_slope_and_volatility(weekly_orders)
        spend_slope, spend_cv = weekly_slope_and_volatility(weekly_spend)
        country = g["Country"].mode().iat[0] if len(g["Country"].mode()) else "Unknown"
        churn = 0 if cust_id in future_active_ids else 1
        rows.append({
            "CustomerID": cust_id,
            "cutoff_date": cutoff.date().isoformat(),
            "Recency": recency_days,
            "TenureDays": tenure_days,
            "Frequency": frequency_total,
            "Monetary": monetary_total,
            "AverageOrderValue": avg_order_value,
            "UniqueProducts": unique_products,
            "AvgBasketSize": avg_basket_size,
            "Orders_30d": orders_30d,
            "Orders_Prev30d": orders_prev_30d,
            "Spend_30d": spend_30d,
            "Spend_Prev30d": spend_prev_30d,
            "Orders_90d": orders_90d,
            "Orders_Prev90d": orders_prev_90d,
            "Spend_90d": spend_90d,
            "Spend_Prev90d": spend_prev_90d,
            "FreqTrendRatio": freq_trend_ratio,
            "SpendTrendRatio": spend_trend_ratio,
            "SpendChange30d": spend_change_30d,
            "FreqChange90d": freq_change_90d,
            "ActivitySlope": activity_slope,
            "ActivityVolatility": activity_cv,
            "SpendSlope": spend_slope,
            "SpendVolatility": spend_cv,
            "RecencyXMonetary": recency_days * np.log1p(monetary_total),
            "RecencyOverTenure": recency_days / max(tenure_days, 1),
            "Country": country,
            "Churn": churn,
        })
    return pd.DataFrame(rows)


def add_rfm_scores(snap):
    # quintile-based RFM score, computed independently *within* each
    # cutoff_date snapshot so it never uses information from other
    # snapshots / the future.
    out = []
    for cutoff, g in snap.groupby("cutoff_date"):
        g = g.copy()
        # lower recency = better -> reverse-rank
        g["R_score"] = pd.qcut(g["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
        g["F_score"] = pd.qcut(g["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        g["M_score"] = pd.qcut(g["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        g["RFM_Score"] = g["R_score"] + g["F_score"] + g["M_score"]
        out.append(g)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    df = load_transactions()
    print(f"Loaded {len(df):,} purchase lines, {df['CustomerID'].nunique():,} customers, "
          f"{df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")
    all_snaps = []
    for cutoff in CUTOFFS:
        snap = build_snapshot(df, cutoff)
        print(f"  cutoff {cutoff}: {len(snap):,} active-customer snapshots, "
              f"churn rate {snap['Churn'].mean():.1%}")
        all_snaps.append(snap)
    full = pd.concat(all_snaps, ignore_index=True)
    full = add_rfm_scores(full)
    full.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(full):,} total snapshot rows to {OUT_PATH}")
    print(f"Overall churn rate: {full['Churn'].mean():.1%}")
    print(f"Unique customers across all snapshots: {full['CustomerID'].nunique():,}")
