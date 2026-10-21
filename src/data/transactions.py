"""
Transaction-ledger support for CuStorA.

Most companies do not keep a tidy one-row-per-customer table with a Churn
column. What they do have is a ledger: one row per order / invoice line with
a customer id, a date and an amount. This module turns such a ledger into
everything the rest of the pipeline needs:

1. detect_columns()        guess which columns are the id / date / amount
2. prepare_transactions()  clean the ledger into a standard 4-column frame
3. build_snapshot()        customer features "as of" any date, using only
                           information that existed on that date
4. build_training_set()    many snapshots, each labelled by what happened in
                           the following `window_days` (the churn label)
5. to_canonical_table()    the customer table the existing stages expect

Churn is DEFINED here, not looked up: a customer who was active before the
cutoff date has churned if they made no purchase in the `window_days` after
it. Because every feature only uses transactions up to the cutoff, and the
label only uses transactions after it, there is no leakage between the two.

Fields a ledger cannot supply (Age, Gender, Complaints, DiscountUsage) are NOT
invented: they are reported as unavailable (see coverage_report) and filled
with neutral placeholders that carry no information.
"""

import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

_SYNONYMS = {
    "customer_id": [
        "customerid", "customer_id", "custid", "cust_id", "clientid", "client_id",
        "userid", "user_id", "accountid", "account_id", "customer", "customerno",
        "customer_number", "memberid", "member_id", "buyerid",
    ],
    "date": [
        "invoicedate", "invoice_date", "orderdate", "order_date", "date",
        "transactiondate", "transaction_date", "purchasedate", "purchase_date",
        "timestamp", "datetime", "createdat", "created_at", "saledate", "sale_date",
        "orderdatetime", "order_datetime",
    ],
    "amount": [
        "amount", "total", "totalamount", "total_amount", "revenue", "sales",
        "value", "price_total", "linetotal", "line_total", "ordertotal",
        "order_total", "spend", "totalprice", "total_price", "netamount",
        "net_amount", "grandtotal", "grand_total", "payment", "paid",
    ],
    "quantity": ["quantity", "qty", "units", "quantityordered"],
    "unit_price": ["unitprice", "unit_price", "price", "priceeach", "itemprice"],
    "invoice": [
        "invoiceno", "invoice_no", "invoice", "invoiceid", "invoice_id", "orderid",
        "order_id", "orderno", "order_no", "transactionid", "transaction_id",
        "receipt", "receiptid", "ordernumber", "order_number", "basketid",
    ],
}

# What each role means, shown on the mapping screen.
ROLE_LABELS = {
    "customer_id": "Customer ID",
    "date": "Purchase date",
    "amount": "Amount (line or order total)",
    "quantity": "Quantity (if no amount column)",
    "unit_price": "Unit price (if no amount column)",
    "invoice": "Order / invoice number (optional)",
}


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_NAME_HINTS = {
    "customer_id": ["cust", "client", "member", "buyer", "account", "userid", "user_id"],
    "amount": ["amount", "total", "revenue", "sales", "spend", "value"],
    "invoice": ["invoice", "order", "receipt", "basket", "transaction"],
}


def _mostly_dates(series):
    sample = series.dropna().astype(str).head(200)
    if sample.empty or sample.str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.5:
        return False  # plain numbers are not dates
    return bool(pd.to_datetime(sample, errors="coerce").notna().mean() >= 0.9)


def detect_columns(columns, df=None):
    """
    Guess {role: column name or None}. Exact known names first, then name
    hints ("cust_no", "total_paid"), then -- when the data itself is given --
    any column whose values are mostly dates. These are only defaults: the
    user confirms or changes every choice on the mapping screen.
    """
    columns = list(columns)
    normalised = {_norm(c): c for c in columns}
    found = {}
    for role, names in _SYNONYMS.items():
        found[role] = None
        for candidate in names:
            key = _norm(candidate)
            if key in normalised:
                found[role] = normalised[key]
                break
    used = {v for v in found.values() if v}
    for role, hints in _NAME_HINTS.items():
        if found[role]:
            continue
        for column in columns:
            if column not in used and any(h in _norm(column) for h in hints):
                found[role] = column
                used.add(column)
                break
    if df is not None and not found["date"]:
        for column in columns:
            if column not in used and _mostly_dates(df[column]):
                found["date"] = column
                break
    return found


def looks_like_transactions(df, mapping=None):
    """
    True when the file is probably a ledger rather than a customer table:
    an id + date + a way to get an amount are present, and customers repeat
    across rows (a customer table has one row per customer).
    """
    if df is None or df.empty:
        return False
    mapping = mapping or detect_columns(df.columns, df)
    has_amount = bool(mapping.get("amount")) or (
        bool(mapping.get("quantity")) and bool(mapping.get("unit_price"))
    )
    if not (mapping.get("customer_id") and mapping.get("date") and has_amount):
        return False
    ids = df[mapping["customer_id"]].dropna()
    if ids.empty:
        return False
    rows_per_customer = len(ids) / max(ids.nunique(), 1)
    return rows_per_customer >= 1.5


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


# Ids that mean "we do not know who this is" rather than a real customer.
_PLACEHOLDER_TEXT = {
    "", "nan", "none", "null", "na", "n/a", "#n/a", "unknown", "guest", "anonymous",
    "missing", "not available", "-", "(blank)", "nat",
}
_PLACEHOLDER_NUMERIC = {"0", "-1", "0.0", "-1.0"}


def anonymous_id_mask(ids, invoices):
    """
    True for rows whose customer id is a placeholder for "unknown customer".
    Text placeholders ("guest", "unknown", ...) always count. Numeric ones
    ("0", "-1") only count when that single id accounts for an implausible
    amount of activity -- many datasets legitimately number customers from 0.
    """
    key = ids.astype(str).str.strip().str.lower()
    mask = key.isin(_PLACEHOLDER_TEXT)
    numeric = key.isin(_PLACEHOLDER_NUMERIC)
    if numeric.any():
        orders_per_id = invoices.groupby(ids).nunique()
        typical = max(float(orders_per_id.median()), 1.0)
        for value in ids[numeric].unique():
            n_orders = int(orders_per_id.get(value, 0))
            if n_orders >= 50 and n_orders >= 25 * typical:
                mask = mask | (ids == value)
    return mask


def prepare_transactions(df, mapping):
    """
    Return (tx, report). `tx` has exactly: CustomerID, Date, Amount, Invoice,
    IsReturn. `report` counts what was dropped (missing values, zero amounts,
    anonymous guest ids) so nothing is hidden.
    """
    cid = mapping.get("customer_id")
    date = mapping.get("date")
    if not cid or not date:
        raise ValueError("A customer id column and a date column are required.")
    amount_col = mapping.get("amount")
    qty, price = mapping.get("quantity"), mapping.get("unit_price")
    if not amount_col and not (qty and price):
        raise ValueError("Pick an amount column, or both a quantity and a unit price column.")
    for column in [cid, date, amount_col, qty, price, mapping.get("invoice")]:
        if column and column not in df.columns:
            raise ValueError(f"Column '{column}' was not found in the file.")

    report = {"rows_in": int(len(df))}
    tx = pd.DataFrame({"CustomerID": df[cid]})
    tx["Date"] = pd.to_datetime(df[date], errors="coerce")
    if amount_col:
        tx["Amount"] = pd.to_numeric(df[amount_col], errors="coerce")
    else:
        tx["Amount"] = pd.to_numeric(df[qty], errors="coerce") * pd.to_numeric(
            df[price], errors="coerce"
        )
    invoice_col = mapping.get("invoice")
    if invoice_col:
        tx["Invoice"] = df[invoice_col].astype("string")
    else:
        # No order number: treat each (customer, day) as one order.
        tx["Invoice"] = (
            tx["CustomerID"].astype("string") + "|" + tx["Date"].dt.strftime("%Y-%m-%d").fillna("")
        )

    bad = tx["CustomerID"].isna() | tx["Date"].isna() | tx["Amount"].isna()
    report["rows_dropped_missing"] = int(bad.sum())
    tx = tx.loc[~bad].copy()

    # Customer ids like 17850.0 (float because of blanks) -> "17850"
    ids = tx["CustomerID"]
    if pd.api.types.is_float_dtype(ids) and (ids % 1 == 0).all():
        ids = ids.astype("int64")
    tx["CustomerID"] = ids.astype(str).str.strip()

    # Guest checkouts are often exported with a placeholder id ("0", "-1",
    # "guest", ...). Summed into ONE row they look like a single enormous
    # customer who buys every day and never leaves -- which wrecks both the
    # model and the customer list -- so they are removed and counted.
    anonymous = anonymous_id_mask(tx["CustomerID"], tx["Invoice"])
    report["rows_dropped_anonymous"] = int(anonymous.sum())
    report["anonymous_ids"] = sorted(tx.loc[anonymous, "CustomerID"].unique().tolist())[:5]
    tx = tx.loc[~anonymous].copy()

    # Invoice numbers like "C536379" mark cancellations (UCI Online Retail and
    # many ERP exports). Only trust that when the file really has an invoice
    # column -- the invoice ids made up above start with the customer id.
    if invoice_col:
        cancelled = tx["Invoice"].fillna("").str.match(r"^[Cc]\d")
    else:
        cancelled = pd.Series(False, index=tx.index)
    tx["IsReturn"] = (tx["Amount"] < 0) | cancelled
    report["return_rows"] = int(tx["IsReturn"].sum())
    zero = tx["Amount"] == 0
    report["rows_dropped_zero_amount"] = int(zero.sum())
    tx = tx.loc[~zero].copy()

    tx["Amount"] = tx["Amount"].abs()
    tx["Date"] = tx["Date"].dt.normalize()
    tx = tx.sort_values("Date").reset_index(drop=True)
    report["rows_used"] = int(len(tx))
    report["customers"] = int(tx["CustomerID"].nunique())
    if tx.empty:
        raise ValueError("No usable rows remain after cleaning. Check the column choices.")
    report["first_date"] = tx["Date"].min().strftime("%Y-%m-%d")
    report["last_date"] = tx["Date"].max().strftime("%Y-%m-%d")
    report["span_days"] = int((tx["Date"].max() - tx["Date"].min()).days)
    return tx, report


# ---------------------------------------------------------------------------
# Snapshot features
# ---------------------------------------------------------------------------

# Features are grouped so their value can be measured (ablation) rather than
# assumed. All of them are computed as of the snapshot date from earlier
# transactions only.
FEATURE_GROUPS = {
    # how long, how much, how often -- the classic RFM view
    "level": [
        "Recency", "TenureDays", "Frequency", "Monetary", "AverageOrderValue",
    ],
    # what the last few weeks look like compared with the weeks before
    "momentum": [
        "Orders_14d", "Orders_30d", "Orders_Prev30d", "Orders_90d", "Orders_Prev90d",
        "Orders_180d", "Spend_90d", "Spend_Prev90d", "Spend_180d",
        "FreqTrendRatio", "SpendTrendRatio", "RecentOrderShare", "RecentSpendShare",
        "ActiveWeeks_90d",
    ],
    # the customer's OWN buying rhythm, and whether they are overdue against it
    "rhythm": [
        "AvgGapDays", "MedianGapDays", "GapStdDays", "LastGapDays", "MaxGapDays",
        "RecencyOverGap", "RecencyOverMedianGap", "OverdueZ",
    ],
    # what a typical basket looks like and whether the latest one was unusual
    "basket": [
        "LastOrderValue", "LastOrderRatio", "AvgLinesPerOrder", "RecentAOVRatio",
    ],
    # BG/NBD "is this customer still alive?" model (see src/churn/bgnbd.py)
    "alive_model": ["PAlive", "ExpectedOrders"],
    "service": ["ReturnRate"],
}
TEMPORAL_FEATURES = [name for group in FEATURE_GROUPS.values() for name in group]

# The model that shipped before the feature upgrade, kept so the improvement
# can be measured against it on the same data and the same split.
LEGACY_FEATURES = [
    "Recency", "TenureDays", "Frequency", "Monetary", "AverageOrderValue",
    "Orders_30d", "Orders_Prev30d", "Orders_90d", "Orders_Prev90d",
    "Spend_90d", "Spend_Prev90d", "FreqTrendRatio", "SpendTrendRatio",
    "AvgGapDays", "RecencyOverGap", "ReturnRate",
]


def to_orders(tx):
    """
    Collapse the line-level ledger to one row per order (customer, invoice, day):
    CustomerID, Invoice, Date, IsReturn, Amount (sum of lines), Lines (line count).
    Every snapshot is computed from this much smaller table.
    """
    if "Lines" in tx.columns:  # already an order table
        return tx
    grouped = tx.groupby(["CustomerID", "Invoice", "Date", "IsReturn"], sort=False)["Amount"]
    orders = grouped.agg(Amount="sum", Lines="size").reset_index()
    return orders.sort_values("Date").reset_index(drop=True)


def _per_customer(frame, column, how):
    return getattr(frame.groupby("CustomerID")[column], how)()


def build_snapshot(tx, as_of, active_days=None, orders=None, horizon_days=90):
    """
    One row per customer, using ONLY transactions dated <= as_of.
    `active_days`: keep only customers who purchased within that many days
    before as_of (the population a retention team can still act on).
    `orders`: the table from to_orders(tx), passed in to avoid recomputing it
    for every snapshot. `horizon_days`: look-ahead used for ExpectedOrders.
    """
    from src.churn import bgnbd

    as_of = pd.Timestamp(as_of).normalize()
    orders = to_orders(tx) if orders is None else orders
    past = orders.loc[orders["Date"] <= as_of]
    sales = past.loc[~past["IsReturn"]]
    if sales.empty:
        return pd.DataFrame()

    grouped = sales.groupby("CustomerID")
    first = grouped["Date"].min()
    last = grouped["Date"].max()
    n_orders = grouped.size()
    spend = grouped["Amount"].sum()
    out = pd.DataFrame(
        {
            "RegistrationDate": first,
            "LastPurchaseDate": last,
            "TotalPurchases": n_orders,
            "TotalSpent": spend.round(2),
        }
    )
    out["Recency"] = (as_of - out["LastPurchaseDate"]).dt.days
    out["TenureDays"] = (as_of - out["RegistrationDate"]).dt.days
    out["Frequency"] = out["TotalPurchases"]
    out["Monetary"] = out["TotalSpent"]
    out["AverageOrderValue"] = (out["TotalSpent"] / out["TotalPurchases"].clip(lower=1)).round(2)

    # ---- momentum: recent windows vs the window before ----
    age = (as_of - sales["Date"]).dt.days.to_numpy()

    def window(lo, hi):
        sub = sales.loc[(age >= lo) & (age < hi)].groupby("CustomerID")
        return sub.size().reindex(out.index).fillna(0), sub["Amount"].sum().reindex(out.index).fillna(0)

    for label, lo, hi in [("14d", 0, 14), ("30d", 0, 30), ("Prev30d", 30, 60),
                          ("90d", 0, 90), ("Prev90d", 90, 180), ("180d", 0, 180)]:
        n, s = window(lo, hi)
        out[f"Orders_{label}"] = n
        if label in ("90d", "Prev90d", "180d"):
            out[f"Spend_{label}"] = s.round(2)
    out["FreqTrendRatio"] = (out["Orders_90d"] / (out["Orders_Prev90d"] + 1)).round(4)
    out["SpendTrendRatio"] = (out["Spend_90d"] / (out["Spend_Prev90d"] + 1)).round(4)
    out["RecentOrderShare"] = (out["Orders_90d"] / out["Frequency"].clip(lower=1)).round(4)
    out["RecentSpendShare"] = (out["Spend_90d"] / out["Monetary"].clip(lower=1e-9)).round(4)
    weeks = (age // 7)
    recent = sales.loc[age < 90, "CustomerID"].to_frame().assign(week=weeks[age < 90])
    out["ActiveWeeks_90d"] = (
        recent.drop_duplicates().groupby("CustomerID").size().reindex(out.index).fillna(0)
    )

    # ---- rhythm: the customer's own gaps between purchase days ----
    days = sales[["CustomerID", "Date"]].drop_duplicates().sort_values(["CustomerID", "Date"])
    gap = days.groupby("CustomerID")["Date"].diff().dt.days
    gap_by_customer = gap.groupby(days["CustomerID"])
    order_days = days.groupby("CustomerID").size().reindex(out.index)
    avg_gap = gap_by_customer.mean().reindex(out.index)
    fallback = out["TenureDays"].clip(lower=1)  # single-order customers: no rhythm known
    out["AvgGapDays"] = avg_gap.fillna(fallback).round(2).clip(lower=1)
    out["MedianGapDays"] = gap_by_customer.median().reindex(out.index).fillna(out["AvgGapDays"]).clip(lower=1)
    out["GapStdDays"] = gap_by_customer.std().reindex(out.index).fillna(0.0)
    out["LastGapDays"] = gap_by_customer.last().reindex(out.index).fillna(out["AvgGapDays"])
    out["MaxGapDays"] = gap_by_customer.max().reindex(out.index).fillna(out["AvgGapDays"])
    out["RecencyOverGap"] = (out["Recency"] / out["AvgGapDays"]).round(4)
    out["RecencyOverMedianGap"] = (out["Recency"] / out["MedianGapDays"]).round(4)
    out["OverdueZ"] = ((out["Recency"] - out["AvgGapDays"]) / (out["GapStdDays"] + 7.0)).round(4)

    # ---- basket ----
    latest = sales.sort_values("Date").groupby("CustomerID").tail(1).set_index("CustomerID")
    out["LastOrderValue"] = latest["Amount"].reindex(out.index)
    out["LastOrderRatio"] = (out["LastOrderValue"] / out["AverageOrderValue"].clip(lower=1e-9)).round(4)
    out["AvgLinesPerOrder"] = grouped["Lines"].mean().round(3)
    recent_aov = (out["Spend_90d"] / out["Orders_90d"].clip(lower=1)).where(out["Orders_90d"] > 0)
    out["RecentAOVRatio"] = (recent_aov / out["AverageOrderValue"].clip(lower=1e-9)).fillna(0).round(4)

    # ---- BG/NBD "still alive?" ----
    x, t_x, T = bgnbd.customer_inputs(first.reindex(out.index), last.reindex(out.index), order_days, as_of)
    params = bgnbd.fit(x, t_x, T)
    out["PAlive"] = bgnbd.p_alive(params, x, t_x, T).round(5)
    out["ExpectedOrders"] = bgnbd.expected_orders(params, x, t_x, T, horizon_days / 7.0).round(5)

    # ---- service ----
    ret = past.loc[past["IsReturn"]].groupby("CustomerID").size()
    out["Returns"] = ret.reindex(out.index).fillna(0)
    out["ReturnRate"] = (out["Returns"] / out["TotalPurchases"].clip(lower=1)).round(4)

    if active_days is not None:
        out = out.loc[out["Recency"] <= active_days]
    out = out.reset_index().rename(columns={"index": "CustomerID"})
    out["SnapshotDate"] = as_of
    return out


def label_snapshot(snapshot, tx, as_of, window_days):
    """Churn = 1 when the customer made NO purchase in (as_of, as_of+window]."""
    as_of = pd.Timestamp(as_of).normalize()
    end = as_of + pd.Timedelta(days=window_days)
    future = tx.loc[(tx["Date"] > as_of) & (tx["Date"] <= end) & (~tx["IsReturn"])]
    returned = set(future["CustomerID"].unique())
    snapshot = snapshot.copy()
    snapshot["Churn"] = (~snapshot["CustomerID"].isin(returned)).astype(int)
    return snapshot


def plan_cutoffs(tx, window_days, step_days=14, min_history_days=30):
    """
    Cutoff dates for training snapshots, newest first. The newest cutoff is
    exactly `window_days` before the end of the data so its label window is
    fully observed. Returns [] when there is not enough history.
    """
    last = tx["Date"].max()
    first = tx["Date"].min()
    newest = last - pd.Timedelta(days=window_days)
    cutoffs = []
    c = newest
    while (c - first).days >= min_history_days:
        cutoffs.append(c)
        c = c - pd.Timedelta(days=step_days)
    return cutoffs


def build_training_set(tx, window_days=90, active_days=90, step_days=14, min_history_days=30, max_cutoffs=30):
    """
    Labelled snapshots over several cutoffs -> one long DataFrame with
    SnapshotDate + Churn. `active_days` = who counts as an active customer.
    Cutoffs are two weeks apart by default (weekly ones were tested and did not
    help: neighbouring snapshots of the same customer are near-duplicates).
    `max_cutoffs` keeps very long ledgers from producing huge training tables.
    """
    cutoffs = plan_cutoffs(tx, window_days, step_days, min_history_days)[:max_cutoffs]
    orders = to_orders(tx)
    frames = []
    for cutoff in cutoffs:
        snap = build_snapshot(
            tx, cutoff, active_days=active_days, orders=orders, horizon_days=window_days
        )
        if snap.empty:
            continue
        frames.append(label_snapshot(snap, orders, cutoff, window_days))
    if not frames:
        return pd.DataFrame(), cutoffs
    return pd.concat(frames, ignore_index=True), cutoffs


# ---------------------------------------------------------------------------
# Canonical table for the existing pipeline stages
# ---------------------------------------------------------------------------

CANONICAL_COLUMNS = [
    "CustomerID", "Age", "Gender", "RegistrationDate", "LastPurchaseDate",
    "TotalPurchases", "TotalSpent", "AverageOrderValue", "EngagementScore",
    "Complaints", "Returns", "DiscountUsage", "Churn",
]


def to_canonical_table(snapshot):
    """
    Map a snapshot onto the 13-column customer table. Churn is left empty
    (it is what we are about to predict). EngagementScore is DERIVED from
    purchase activity: 0-100 percentile of orders in the last 90 days.
    """
    s = snapshot.copy()
    out = pd.DataFrame({"CustomerID": s["CustomerID"]})
    out["Age"] = np.nan  # not in a ledger
    out["Gender"] = "Unknown"  # not in a ledger
    out["RegistrationDate"] = s["RegistrationDate"].dt.strftime("%Y-%m-%d")
    out["LastPurchaseDate"] = s["LastPurchaseDate"].dt.strftime("%Y-%m-%d")
    out["TotalPurchases"] = s["TotalPurchases"]
    out["TotalSpent"] = s["TotalSpent"]
    out["AverageOrderValue"] = s["AverageOrderValue"]
    out["EngagementScore"] = (s["Orders_90d"].rank(pct=True, method="average") * 100).round(1)
    out["Complaints"] = 0  # not in a ledger
    out["Returns"] = s["Returns"]
    out["DiscountUsage"] = 0  # not in a ledger
    out["Churn"] = np.nan
    return out[CANONICAL_COLUMNS]


def coverage_report(tx_report, tx):
    """What each CuStorA field is backed by for this upload (shown in the UI)."""
    has_returns = bool(tx_report.get("return_rows", 0))
    rows = [
        ("CustomerID", "Observed", "Your customer id column"),
        ("RegistrationDate", "Derived", "Date of the customer's first purchase"),
        ("LastPurchaseDate", "Observed", "Date of the customer's latest purchase"),
        ("TotalPurchases", "Derived", "Count of distinct orders"),
        ("TotalSpent", "Derived", "Sum of purchase amounts"),
        ("AverageOrderValue", "Derived", "Total spent / orders"),
        ("Returns", "Observed" if has_returns else "Not available",
         "Cancelled / negative-amount orders" if has_returns else "No returns found in the file"),
        ("EngagementScore", "Derived", "Percentile of orders in the last 90 days (not a survey score)"),
        ("Churn", "Derived", "No purchase in the window after a cutoff date (computed, never guessed)"),
        ("Age", "Not available", "A ledger does not contain it; not used"),
        ("Gender", "Not available", "A ledger does not contain it; not used"),
        ("Complaints", "Not available", "A ledger does not contain it; not used"),
        ("DiscountUsage", "Not available", "A ledger does not contain it; not used"),
    ]
    return pd.DataFrame(rows, columns=["Field", "Status", "How it is obtained"])
