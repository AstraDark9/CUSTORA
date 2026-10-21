"""Transaction-ledger support: detection, leakage-safety, and the forward-in-time churn model."""

import numpy as np
import pandas as pd
import pytest

from src.churn import feature_contract, temporal_churn
from src.clustering.segment_naming import assign_segment_names
from src.data import transactions as t
from src.health import relationship_health as health


def make_ledger(n_customers=700, days=420, seed=0):
    """Fake shop: loyal customers buy every ~3 weeks; leavers fade out at a random date."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2022-01-01")
    rows = []
    for cid in range(n_customers):
        join = int(rng.integers(0, max(days // 3, 1)))
        leaver = rng.random() < 0.45
        stop = int(rng.integers(min(join + 30, days - 1), days)) if leaver else days + 1
        gap = rng.uniform(14, 35)
        day = join
        while day < min(stop, days):
            rows.append((f"C{cid}", start + pd.Timedelta(days=int(day)), round(rng.uniform(10, 90), 2), f"I{cid}-{int(day)}"))
            day += gap * rng.uniform(0.7, 1.4) * (1 + max(0, day - (stop - 80)) / 60 if leaver else 1)
    return pd.DataFrame(rows, columns=["cust", "when", "amount", "order"])


@pytest.fixture(scope="module")
def tx():
    raw = make_ledger()
    mapping = t.detect_columns(raw.columns, raw)
    out, _ = t.prepare_transactions(raw, mapping)
    return out


def test_detect_columns_standard_and_odd_names():
    m = t.detect_columns(["CustomerID", "InvoiceDate", "Quantity", "UnitPrice", "InvoiceNo"])
    assert m["customer_id"] == "CustomerID" and m["date"] == "InvoiceDate" and m["invoice"] == "InvoiceNo"
    raw = make_ledger(50, 100)
    m = t.detect_columns(raw.columns, raw)
    assert m["customer_id"] == "cust" and m["date"] == "when" and m["amount"] == "amount"
    assert t.looks_like_transactions(raw, m)


def test_customer_table_is_not_mistaken_for_ledger():
    table = pd.DataFrame({"CustomerID": range(100), "LastPurchaseDate": ["2024-01-01"] * 100, "TotalSpent": 5.0})
    assert not t.looks_like_transactions(table)


def test_prepare_transactions_handles_returns_and_bad_rows():
    raw = pd.DataFrame({
        "id": [1, 1, 2, None, 3],
        "d": ["2024-01-01", "2024-01-05", "not a date", "2024-01-02", "2024-01-03"],
        "amt": [10, -10, 5, 5, 0],
        "inv": ["A1", "CA2", "A3", "A4", "A5"],
    })
    tx, rep = t.prepare_transactions(raw, {"customer_id": "id", "date": "d", "amount": "amt", "invoice": "inv"})
    assert rep["rows_dropped_missing"] == 2  # bad date, missing id
    assert rep["rows_dropped_zero_amount"] == 1
    assert tx["IsReturn"].sum() == 1 and (tx["Amount"] > 0).all()


def test_snapshot_uses_only_information_up_to_the_cutoff(tx):
    cutoff = tx["Date"].min() + pd.Timedelta(days=200)
    full = t.build_snapshot(tx, cutoff)
    truncated = t.build_snapshot(tx.loc[tx["Date"] <= cutoff], cutoff)
    pd.testing.assert_frame_equal(
        full.sort_values("CustomerID").reset_index(drop=True),
        truncated.sort_values("CustomerID").reset_index(drop=True),
    )


def test_churn_label_looks_only_at_the_window_after_the_cutoff():
    raw = pd.DataFrame({
        "c": ["a", "a", "b", "b", "c"],
        "d": ["2024-01-01", "2024-03-01", "2024-01-02", "2024-06-30", "2024-01-03"],
        "x": [10] * 5,
    })
    tx, _ = t.prepare_transactions(raw, {"customer_id": "c", "date": "d", "amount": "x"})
    cutoff = pd.Timestamp("2024-02-01")
    snap = t.label_snapshot(t.build_snapshot(tx, cutoff), tx, cutoff, 90)
    churn = dict(zip(snap["CustomerID"], snap["Churn"]))
    assert churn == {"a": 0, "b": 1, "c": 1}  # a buys in the window; b only much later; c never


def test_too_little_history_is_refused_clearly():
    raw = make_ledger(200, 100)
    tx, _ = t.prepare_transactions(raw, t.detect_columns(raw.columns, raw))
    with pytest.raises(ValueError, match="history"):
        temporal_churn.run(tx, window_days=90, model_dir="/tmp/_m", report_dir="/tmp/_r")


def test_forward_model_beats_chance_and_is_tested_on_the_newest_period(tx, tmp_path):
    out = temporal_churn.run(tx, window_days=90, active_days=90,
                             model_dir=str(tmp_path / "m"), report_dir=str(tmp_path / "r"))
    s = out["summary"]
    best = s["test"][s["selected_model"]]
    assert best["roc_auc"] > 0.62
    # train < validation < test in time, and train labels end before the test snapshot
    assert max(s["split"]["train_cutoffs"]) < s["split"]["validation_cutoff"] < s["split"]["test_cutoff"]
    gap = pd.Timestamp(s["split"]["test_cutoff"]) - pd.Timestamp(s["split"]["validation_cutoff"])
    assert gap.days >= 90
    for name in ["decision_thresholds.json", "backtest_summary.json", "final_test_evaluation.csv", "test_roc_curve.png"]:
        assert (tmp_path / "r" / name).exists()


def test_downstream_stages_ask_the_model_for_its_columns(tx, tmp_path):
    out = temporal_churn.run(tx, window_days=90, model_dir=str(tmp_path / "m"), report_dir=str(tmp_path / "r"))
    import joblib
    model = joblib.load(out["best_model_path"])
    assert feature_contract.model_input_columns(model) == t.TEMPORAL_FEATURES
    wide = t.build_snapshot(tx, tx["Date"].max()).assign(Extra=1, Churn=np.nan)
    X = feature_contract.select_model_features(wide, ["Churn"], model)
    assert list(X.columns) == t.TEMPORAL_FEATURES
    assert model.predict_proba(X).shape[1] == 2


def test_health_does_not_hand_out_free_points_when_friction_is_unobserved():
    df = pd.DataFrame({
        "Recency": [0, 300], "EngagementScore": [100, 0], "Complaints": [0, 0], "Returns": [0, 0],
    })
    scored = health.compute_relationship_health(df)
    assert scored["RelationshipHealthScore"].iloc[0] > 95
    assert scored["RelationshipHealthScore"].iloc[1] < 10  # was ~25 when friction counted as "perfect"


def test_segment_names_survive_one_huge_outlier_cluster():
    rng = np.random.default_rng(1)
    n = 300
    data = pd.DataFrame({
        "Cluster": np.repeat([0, 1, 2], n // 3),
        "Monetary": np.concatenate([rng.normal(200, 40, 100), rng.normal(900, 100, 100), rng.normal(90000, 5000, 100)]),
        "Frequency": np.concatenate([rng.normal(2, .5, 100), rng.normal(8, 1, 100), rng.normal(70, 5, 100)]),
        "EngagementScore": np.concatenate([rng.normal(30, 5, 100), rng.normal(70, 5, 100), rng.normal(95, 2, 100)]),
        "Recency": np.concatenate([rng.normal(60, 5, 100), rng.normal(20, 5, 100), rng.normal(5, 1, 100)]),
    })
    names = assign_segment_names(data).groupby("Cluster")["SegmentName"].first()
    assert names.nunique() == 3
    assert "High-Value" in names[2] and "High-Value" not in names[0]
