"""Guest-id cleaning, BG/NBD, the richer features, and the honest-evaluation guarantees."""

import numpy as np
import pandas as pd
import pytest

from src.churn import bgnbd, temporal_churn
from src.data import transactions as t
from src.explainability.reason_translator import build_customer_reasons
from tests.test_transactions import make_ledger


@pytest.fixture(scope="module")
def tx():
    raw = make_ledger()
    out, _ = t.prepare_transactions(raw, t.detect_columns(raw.columns, raw))
    return out


# ---------- anonymous guest ids ----------

def _ledger_with_guest(guest_id):
    base = make_ledger(60, 200)
    rng = np.random.default_rng(1)
    guest = pd.DataFrame({
        "cust": guest_id,
        "when": pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 200, 400), unit="D"),
        "amount": 20.0,
        "order": [f"G{i}" for i in range(400)],
    })
    return pd.concat([base, guest], ignore_index=True)


def test_guest_placeholder_id_is_removed_and_reported():
    raw = _ledger_with_guest(0)
    tx, rep = t.prepare_transactions(raw, t.detect_columns(raw.columns, raw))
    assert rep["rows_dropped_anonymous"] == 400 and rep["anonymous_ids"] == ["0"]
    assert "0" not in set(tx["CustomerID"])


def test_text_placeholders_always_count_as_anonymous():
    raw = _ledger_with_guest("Guest")
    tx, rep = t.prepare_transactions(raw, t.detect_columns(raw.columns, raw))
    assert rep["rows_dropped_anonymous"] == 400


def test_a_real_customer_numbered_zero_is_kept():
    """Many systems number customers from 0; a normal-sized customer 0 must survive."""
    raw = make_ledger(60, 200)
    raw["cust"] = raw["cust"].str.replace("C", "").astype(int)  # ids 0..59
    tx, rep = t.prepare_transactions(raw, t.detect_columns(raw.columns, raw))
    assert rep["rows_dropped_anonymous"] == 0 and "0" in set(tx["CustomerID"])


# ---------- BG/NBD ----------

def test_bgnbd_recovers_sensible_behaviour():
    rng = np.random.default_rng(3)
    n, T = 3000, 52.0
    x, t_x = np.zeros(n), np.zeros(n)
    for i in range(n):
        rate, drop = rng.gamma(2, 0.2), rng.beta(0.8, 3)
        time, alive = 0.0, True
        while alive:
            time += rng.exponential(1 / rate)
            if time > T:
                break
            x[i] += 1
            t_x[i] = time
            alive = rng.random() > drop
    params = bgnbd.fit(x, t_x, np.full(n, T))
    assert all(p > 0 for p in params)
    alive = bgnbd.p_alive(params, x, t_x, np.full(n, T))
    assert alive.min() >= 0 and alive.max() <= 1
    silent = bgnbd.p_alive(params, np.array([8.0]), np.array([10.0]), np.array([50.0]))[0]
    recent = bgnbd.p_alive(params, np.array([8.0]), np.array([49.0]), np.array([50.0]))[0]
    assert silent < recent  # same history, but silent for 40 weeks: less likely still alive
    expected = bgnbd.expected_orders(params, x[:50], t_x[:50], np.full(50, T), 13.0)
    assert np.all(np.isfinite(expected)) and np.all(expected >= 0)


def test_bgnbd_falls_back_instead_of_crashing_on_tiny_data():
    assert bgnbd.fit(np.array([1.0]), np.array([1.0]), np.array([2.0])) == bgnbd.DEFAULT_PARAMS


# ---------- features ----------

def test_snapshot_features_ignore_everything_after_the_cutoff(tx):
    cutoff = tx["Date"].min() + pd.Timedelta(days=200)
    clean = t.build_snapshot(tx, cutoff, active_days=90)
    future = tx.loc[tx["Date"] > cutoff].copy()
    poisoned = pd.concat([tx, future.assign(Amount=future["Amount"] * 1000, Invoice="X" + future["Invoice"])])
    again = t.build_snapshot(poisoned, cutoff, active_days=90)
    pd.testing.assert_frame_equal(
        clean.sort_values("CustomerID").reset_index(drop=True),
        again.sort_values("CustomerID").reset_index(drop=True),
    )


def test_every_declared_feature_exists_and_is_finite_or_nan(tx):
    snap = t.build_snapshot(tx, tx["Date"].max() - pd.Timedelta(days=90), active_days=90)
    assert set(t.TEMPORAL_FEATURES) <= set(snap.columns)
    values = snap[t.TEMPORAL_FEATURES].to_numpy(dtype=float)
    assert not np.isinf(values).any()
    assert snap["PAlive"].between(0, 1).all()


def test_legacy_features_are_a_subset_so_the_comparison_is_like_for_like():
    assert set(t.LEGACY_FEATURES) <= set(t.TEMPORAL_FEATURES)


# ---------- evaluation guarantees ----------

@pytest.fixture(scope="module")
def trained(tx, tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    return temporal_churn.run(tx, model_dir=str(d / "m"), report_dir=str(d / "r"))["summary"]


def test_validation_is_embargoed_from_training_when_history_allows(trained):
    s = trained["split"]
    assert s["validation_embargo"] == "strict"
    newest_train = max(pd.Timestamp(c) for c in s["train_cutoffs"])
    oldest_val = min(pd.Timestamp(c) for c in s["validation_cutoffs"])
    assert (oldest_val - newest_train).days >= trained["window_days"]  # outcomes known before validation


def test_test_period_is_after_validation_by_a_full_window(trained):
    s = trained["split"]
    gap = pd.Timestamp(s["test_cutoff"]) - max(pd.Timestamp(c) for c in s["validation_cutoffs"])
    assert gap.days >= trained["window_days"]


def test_summary_reports_uncertainty_and_the_previous_model(trained):
    best = trained["test"][trained["selected_model"]]
    lo, hi = best["roc_auc_ci95"]
    assert lo <= best["roc_auc"] <= hi
    assert trained["test_previous_model"]["roc_auc"] > 0.5
    assert len(trained["auc_gain_vs_previous_ci95"]) == 2
    assert best["roc_auc"] > trained["test_baseline_recency"]["roc_auc"]


def test_non_selected_models_are_never_scored_on_test(trained):
    assert list(trained["test"]) == [trained["selected_model"]]


def test_balanced_threshold_does_not_flag_everyone_when_half_churn():
    """F1 degenerates to 'flag everybody' at high churn rates; Youden's J must not."""
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.55).astype(int)
    p = np.clip(0.55 + (y - 0.55) * 0.25 + rng.normal(0, 0.15, 4000), 0.01, 0.99)
    best = temporal_churn.select_balanced_threshold(y, p)
    flagged = (p >= best["threshold"]).mean()
    assert 0.3 < flagged < 0.85 and best["youden_j"] > 0.2


def test_calibration_is_strictly_monotone_so_ranking_has_no_ties(tx, trained, tmp_path):
    snaps, cutoffs = t.build_training_set(tx, 90, 90)
    X, y = snaps[t.TEMPORAL_FEATURES], snaps["Churn"].astype(int)
    pipe = temporal_churn.fit_calibrated("rf", X, y, snaps["CustomerID"])
    raw = pipe.named_steps["model"].calibrated_classifiers_[0].estimator.predict_proba(
        pipe.named_steps["preprocessor"].transform(X.iloc[:800]))[:, 1]
    cal = pipe.predict_proba(X.iloc[:800])[:, 1]
    order = np.argsort(raw)
    assert np.all(np.diff(cal[order]) >= -1e-12)


def test_gain_curve_is_monotone_and_ends_at_everyone():
    g = temporal_churn.gain_curve(np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 0]), np.linspace(1, 0, 10))
    assert g["ChurnersReachedPct"].is_monotonic_increasing and g.iloc[-1]["ChurnersReachedPct"] == 100


# ---------- explanations stay truthful ----------

def test_reason_that_contradicts_the_customers_real_value_is_skipped():
    customer = {"Monetary": 168000.0, "Frequency": 2.0}
    shap = [{"Feature": "num__Monetary", "SHAP_Value": 1.5}, {"Feature": "num__Frequency", "SHAP_Value": 0.4}]
    reference = {"Monetary": 900.0, "Frequency": 3.0}
    reasons = build_customer_reasons(customer, shap, reference=reference)
    labels = [r["label"] for r in reasons]
    assert "Total spend" not in " ".join(labels)  # "spend is on the lower end" would be false
    assert all("magnitude" in r for r in reasons)
