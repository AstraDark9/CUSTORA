"""
Forward-looking churn model trained on a transaction ledger.

How the evidence is produced (this is what makes the accuracy claim honest):

* Snapshots are taken at several cutoff dates, two weeks apart. Each snapshot's
  features use only transactions up to its cutoff; its label (Churn = no
  purchase in the next `window_days`) uses only transactions after it.
* TEST = the newest cutoff: the most recent period the model could be judged on.
  Nothing used to choose, fit or calibrate anything sees it, until the very end.
* VALIDATION = the two newest cutoffs whose label windows had ended by the test
  cutoff. They are used to pick the model type and the decision threshold.
* EMBARGO: the model that is scored on validation is trained only on cutoffs
  whose label windows ended before the validation cutoff. Without this, a
  training snapshot of customer X a week earlier carries (nearly) the answer
  for the same customer in the validation snapshot, and validation scores are
  inflated. When the history is too short for that, validation falls back to a
  relaxed split and the report says so.
* After the decisions are frozen, the chosen model is refit on every snapshot
  whose label window ended before the test cutoff, scored ONCE on the test
  snapshot, and finally refit on everything to score customers today.

Alongside the model we score two yardsticks on the same test period: a naive
baseline ("the longer since the last purchase, the likelier to churn") and the
previous model (the earlier feature set and random-forest settings), so what
the upgrade added is measured, with confidence intervals, not asserted.
"""

import json
import os
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.data import transactions

MODEL_DIR = "models/churn"
REPORT_DIR = "reports/churn"
MIN_ROWS_PER_SPLIT = 50
MIN_TRAIN_CUTOFFS = 3
N_VALIDATION_CUTOFFS = 2
BOOTSTRAPS = 400

MODEL_LABELS = {
    "lr": "Logistic Regression",
    "rf": "Random Forest",
    "hgb": "Gradient Boosting",
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _signed_log(values):
    return np.sign(values) * np.log1p(np.abs(values))


class RangeClipper(BaseEstimator, TransformerMixin):
    """
    Clip every feature to the 1st-99th percentile seen in training. A linear model
    extrapolates without limit, so a customer whose tenure is far beyond anything in
    the (older) training snapshots would otherwise get an absurd score; trees are
    flat outside their training range and do not need this.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.nanpercentile(X, 1, axis=0)
        self.high_ = np.nanpercentile(X, 99, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def make_preprocessor(kind, columns):
    """Median-impute, (log-squash and range-clip for the linear model), standardise."""
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if kind == "lr":
        steps.append(("log", FunctionTransformer(_signed_log, feature_names_out="one-to-one")))
        steps.append(("clip", RangeClipper()))
    steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([("num", Pipeline(steps), list(columns))])


def make_estimator(kind):
    if kind == "lr":
        return LogisticRegression(max_iter=3000, C=0.3, random_state=42)
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=50, max_features="sqrt",
            random_state=42, n_jobs=-1,
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.04, max_iter=250, min_samples_leaf=80,
            l2_regularization=5.0, random_state=42,
        )
    raise KeyError(kind)


def candidate_models():
    """Kept for callers that want the list of model keys and labels."""
    return dict(MODEL_LABELS)


def fit_raw(kind, X, y):
    """Uncalibrated pipeline; enough to compare models by how well they RANK."""
    pre = make_preprocessor(kind, X.columns)
    return Pipeline([("preprocessor", pre), ("model", make_estimator(kind))]).fit(X, y)


def fit_calibrated(kind, X, y, groups):
    """
    Pipeline whose probabilities can be read as probabilities.

    Platt (sigmoid) scaling is used rather than isotonic: it is strictly
    increasing, so it can never create ties among customers and the ranking
    the retention team works from is exactly the model's own. Calibration folds
    are split by customer, so the same customer never sits on both sides.
    """
    pre = make_preprocessor(kind, X.columns).fit(X)
    Xt = pre.transform(X)
    groups = np.asarray(groups)
    cv = 3
    if pd.Series(groups).nunique() >= 3:
        cv = list(GroupKFold(n_splits=3).split(Xt, y, groups))
    calibrated = CalibratedClassifierCV(
        estimator=make_estimator(kind), method="sigmoid", cv=cv, ensemble=False
    )
    calibrated.fit(Xt, y)
    return Pipeline([("preprocessor", pre), ("model", calibrated)])


def _positive(pipe, X):
    return pipe.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _lift(y, p, fraction):
    y = np.asarray(y)
    k = max(int(round(len(y) * fraction)), 1)
    top = np.argsort(-np.asarray(p), kind="stable")[:k]
    base = y.mean()
    return float(y[top].mean() / base) if base > 0 else float("nan")


def score_set(y, p, threshold=0.5):
    y = np.asarray(y)
    p = np.asarray(p)
    pred = (p >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "lift_top10pct": round(_lift(y, p, 0.10), 3),
        "lift_top20pct": round(_lift(y, p, 0.20), 3),
        "rows": int(len(y)),
        "churn_rate": round(float(y.mean()), 4),
        "mean_predicted": round(float(p.mean()), 4),
        "flagged_share": round(float(pred.mean()), 4),
    }


def select_balanced_threshold(y, p):
    """
    Probability cutoff that maximises Youden's J (true-positive rate minus
    false-positive rate, i.e. balanced accuracy). Unlike F1 or plain accuracy it
    does not drift with the share of customers who churn, which matters because
    that share moves with the season: F1 picks "flag everybody" when half the
    customers churn, which is how the old cutoff of 0.25 flagged most of a quieter
    test period.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    best = None
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = p >= threshold
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        tpr = pred[y == 1].mean()
        fpr = pred[y == 0].mean()
        row = {
            "threshold": float(threshold),
            "youden_j": float(tpr - fpr),
            "recall": float(tpr),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float((tpr + (1 - fpr)) / 2),
        }
        if best is None or row["youden_j"] > best["youden_j"]:
            best = row
    return best or {"threshold": 0.5, "youden_j": 0.0, "recall": 0.0, "precision": 0.0,
                    "f1": 0.0, "balanced_accuracy": 0.5}


def bootstrap_auc(y, p, p_other=None, n=BOOTSTRAPS, seed=0):
    """95% interval for ROC-AUC, and (if `p_other`) for the paired AUC difference."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    aucs, diffs = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        a = roc_auc_score(y[idx], np.asarray(p)[idx])
        aucs.append(a)
        if p_other is not None:
            diffs.append(a - roc_auc_score(y[idx], np.asarray(p_other)[idx]))
    ci = [round(float(np.quantile(aucs, 0.025)), 4), round(float(np.quantile(aucs, 0.975)), 4)]
    if p_other is None:
        return ci, None
    return ci, [round(float(np.quantile(diffs, 0.025)), 4), round(float(np.quantile(diffs, 0.975)), 4)]


def gain_curve(y, p):
    """Share of all churners found, and hit-rate, when the top x% riskiest customers are contacted."""
    y = np.asarray(y)
    order = np.argsort(-np.asarray(p), kind="stable")
    y_sorted = y[order]
    total = max(int(y.sum()), 1)
    rows = []
    for pct in range(1, 101):
        k = max(int(np.ceil(len(y) * pct / 100)), 1)
        hits = int(y_sorted[:k].sum())
        rows.append({
            "ContactedPct": pct,
            "Contacted": k,
            "ChurnersReachedPct": round(hits / total * 100, 2),
            "HitRatePct": round(hits / k * 100, 2),
            "Lift": round((hits / k) / y.mean(), 3) if y.mean() > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_snapshots(snaps, cutoffs, window_days, n_validation=N_VALIDATION_CUTOFFS):
    """
    Returns a dict with train / val / test frames plus `eligible_train` (every
    snapshot whose label window ended before the test cutoff) and `info`.
    `cutoffs` is newest-first.
    """
    window = pd.Timedelta(days=window_days)
    test_cutoff = cutoffs[0]
    eligible = [c for c in cutoffs[1:] if c <= test_cutoff - window]
    if len(eligible) < 2:
        needed = window_days * 2 + 60 + 30
        raise ValueError(
            "Not enough history to train and fairly test a churn model: "
            f"need roughly {needed}+ days of transactions for a {window_days}-day churn window."
        )
    val_cutoffs = eligible[:n_validation]
    strict_train = [c for c in eligible if c <= min(val_cutoffs) - window]
    embargo = "strict"
    train_cutoffs = strict_train
    if len(strict_train) < MIN_TRAIN_CUTOFFS:
        embargo = "relaxed"
        train_cutoffs = [c for c in eligible if c not in val_cutoffs]
        if len(train_cutoffs) < 2:
            val_cutoffs = eligible[:1]
            train_cutoffs = eligible[1:]
    snap = snaps["SnapshotDate"]
    parts = {
        "train": snaps.loc[snap.isin(train_cutoffs)],
        "val": snaps.loc[snap.isin(val_cutoffs)],
        "test": snaps.loc[snap == test_cutoff],
        "eligible_train": snaps.loc[snap.isin(eligible)],
    }
    for name in ("train", "val", "test"):
        part = parts[name]
        if len(part) < MIN_ROWS_PER_SPLIT or part["Churn"].nunique() < 2:
            raise ValueError(
                f"The {name} period has too few customers (or only one outcome) to evaluate "
                "a churn model. Use a longer history or a shorter churn window."
            )
    fmt = lambda cs: [c.strftime("%Y-%m-%d") for c in cs]
    parts["info"] = {
        "test_cutoff": test_cutoff.strftime("%Y-%m-%d"),
        "validation_cutoff": val_cutoffs[0].strftime("%Y-%m-%d"),
        "validation_cutoffs": fmt(val_cutoffs),
        "train_cutoffs": fmt(train_cutoffs),
        "validation_embargo": embargo,
    }
    return parts


# ---------------------------------------------------------------------------
# Plots (kept for the written report; the app also draws live charts from the JSON)
# ---------------------------------------------------------------------------

BG, FG, GRID = "#0C141B", "#C9D8D3", "#25343F"
MINT, CORAL, AMBER, BLUE = "#34D3AA", "#FF7A59", "#F5B941", "#6C9CFF"


def _style(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _save_plots(report_dir, y_test, curves, baseline_score, deciles):
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    _style(ax, fig)
    for (name, p), color in zip(curves.items(), [MINT, BLUE]):
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC {roc_auc_score(y_test, p):.2f})")
    fpr, tpr, _ = roc_curve(y_test, baseline_score)
    ax.plot(fpr, tpr, "--", color=AMBER, lw=1.6,
            label=f"Baseline: days since last order (AUC {roc_auc_score(y_test, baseline_score):.2f})")
    ax.plot([0, 1], [0, 1], ":", color="#7C8F89", label="Random guessing")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Held-out future period: ROC curve", fontsize=11)
    leg = ax.legend(fontsize=7, loc="lower right", facecolor=BG, edgecolor=GRID)
    for text in leg.get_texts():
        text.set_color(FG)
    fig.tight_layout()
    fig.savefig(f"{report_dir}/test_roc_curve.png", dpi=130, facecolor=BG)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    _style(ax, fig)
    rates = deciles["ActualChurnRate"] * 100
    colors = [CORAL if r > rates.mean() * 1.15 else (AMBER if r > rates.mean() * 0.7 else MINT) for r in rates]
    ax.bar(deciles["RiskDecile"], rates, color=colors, width=0.72)
    ax.axhline(rates.mean(), color=FG, ls="--", lw=1, label="Average churn rate")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Predicted risk decile (1 = highest risk)")
    ax.set_ylabel("Customers who actually churned (%)")
    ax.set_title("Held-out future period: churn by risk decile", fontsize=11)
    leg = ax.legend(fontsize=8, facecolor=BG, edgecolor=GRID)
    for text in leg.get_texts():
        text.set_color(FG)
    fig.tight_layout()
    fig.savefig(f"{report_dir}/test_risk_deciles.png", dpi=130, facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run(
    tx,
    window_days=90,
    active_days=90,
    model_dir=MODEL_DIR,
    report_dir=REPORT_DIR,
    source_label="Uploaded transactions",
    step_days=14,
):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    print("=" * 60)
    print("CuStorA - Forward-looking churn model (transaction data)")
    print("=" * 60)

    snaps, cutoffs = transactions.build_training_set(
        tx, window_days=window_days, active_days=active_days, step_days=step_days
    )
    if snaps.empty or len(cutoffs) < 3:
        raise ValueError(
            "Not enough history to build churn training data. "
            f"Found {int((tx['Date'].max() - tx['Date'].min()).days)} days of transactions; "
            f"a {window_days}-day churn window needs roughly {window_days * 2 + 90}+ days."
        )
    features = list(transactions.TEMPORAL_FEATURES)
    legacy = list(transactions.LEGACY_FEATURES)
    parts = split_snapshots(snaps, cutoffs, window_days)
    train, val, test, eligible = parts["train"], parts["val"], parts["test"], parts["eligible_train"]
    split_info = parts["info"]
    print(f"Snapshots: {len(snaps)} rows over {len(cutoffs)} cutoffs "
          f"(validation embargo: {split_info['validation_embargo']})")
    print(f"Train {len(train)} | Validation {len(val)} | Test {len(test)} (newest period)")
    ytr, yva, yte = (part["Churn"].astype(int).to_numpy() for part in (train, val, test))

    # ---- 1. choose the model type on validation (ranking quality only) ----
    rows, val_raw = [], {}
    for kind, label in MODEL_LABELS.items():
        pipe = fit_raw(kind, train[features], ytr)
        p = _positive(pipe, val[features])
        val_raw[kind] = p
        per_fold = []
        for cutoff in val["SnapshotDate"].unique():
            mask = (val["SnapshotDate"] == cutoff).to_numpy()
            if len(np.unique(yva[mask])) == 2:
                per_fold.append(roc_auc_score(yva[mask], p[mask]))
        m = score_set(yva, p)
        rows.append({
            "Model": label, "Key": kind,
            "ValROC-AUC": round(float(np.mean(per_fold)), 4),
            "ValPR-AUC": m["pr_auc"], "ValLift@10%": m["lift_top10pct"], "ValBrier": m["brier"],
        })
        print(f"  {label}: validation ROC-AUC {np.mean(per_fold):.4f}")
    comparison = pd.DataFrame(rows).sort_values("ValROC-AUC", ascending=False).reset_index(drop=True)
    best_kind = comparison.iloc[0]["Key"]
    best_name = MODEL_LABELS[best_kind]
    comparison["Selected"] = comparison["Key"] == best_kind
    comparison.drop(columns=["Key"]).to_csv(f"{report_dir}/churn_model_comparison.csv", index=False)

    # ---- 2. calibrate the winner and choose the decision threshold on validation ----
    val_model = fit_calibrated(best_kind, train[features], ytr, train["CustomerID"])
    val_prob = _positive(val_model, val[features])
    operating = select_balanced_threshold(yva, val_prob)
    threshold = operating["threshold"]
    print(f"Chosen model: {best_name}; balanced threshold {threshold} "
          f"(validation recall {operating['recall']:.2f}, precision {operating['precision']:.2f})")

    # ---- 3. TEST: touched once, after every decision above ----
    fit_set = eligible
    test_model = fit_calibrated(best_kind, fit_set[features], fit_set["Churn"].astype(int),
                                fit_set["CustomerID"])
    test_prob = _positive(test_model, test[features])
    prev_pipe = _fit_previous(fit_set, legacy)
    prev_prob = _positive(prev_pipe, test[legacy])
    baseline = test["Recency"].to_numpy(dtype=float)

    test_metrics = score_set(yte, test_prob, threshold)
    ci, diff_prev_ci = bootstrap_auc(yte, test_prob, prev_prob)
    _, diff_base_ci = bootstrap_auc(yte, test_prob, baseline)
    test_metrics["roc_auc_ci95"] = ci
    prev_metrics = score_set(yte, prev_prob, threshold)
    prev_metrics["roc_auc_ci95"] = bootstrap_auc(yte, prev_prob)[0]
    base_metrics = {
        "roc_auc": round(float(roc_auc_score(yte, baseline)), 4),
        "pr_auc": round(float(average_precision_score(yte, baseline)), 4),
        "lift_top10pct": round(_lift(yte, baseline, 0.10), 3),
    }
    print(f"TEST ({split_info['test_cutoff']}): {best_name} ROC-AUC {test_metrics['roc_auc']} "
          f"(95% CI {ci[0]}-{ci[1]}) | previous {prev_metrics['roc_auc']} | "
          f"recency baseline {base_metrics['roc_auc']}")

    order = np.argsort(-test_prob, kind="stable")
    decile_idx = np.array_split(order, 10)
    deciles = pd.DataFrame({
        "RiskDecile": range(1, 11),
        "Customers": [len(d) for d in decile_idx],
        "ActualChurnRate": [float(yte[d].mean()) for d in decile_idx],
        "AvgPredictedRisk": [float(test_prob[d].mean()) for d in decile_idx],
    })
    deciles.to_csv(f"{report_dir}/test_risk_deciles.csv", index=False)
    gain = gain_curve(yte, test_prob)
    gain.to_csv(f"{report_dir}/gain_curve.csv", index=False)
    _save_plots(report_dir, yte, {best_name: test_prob, "Previous model": prev_prob}, baseline, deciles)

    pd.DataFrame([{
        "Model": f"{best_name} (calibrated, threshold={threshold})",
        "Accuracy": test_metrics["accuracy"], "Balanced accuracy": test_metrics["balanced_accuracy"],
        "Precision": test_metrics["precision"], "Recall": test_metrics["recall"],
        "F1": test_metrics["f1"], "ROC-AUC": test_metrics["roc_auc"],
        "Brier": test_metrics["brier"], "TestRows": test_metrics["rows"],
    }]).to_csv(f"{report_dir}/final_test_evaluation.csv", index=False)

    # ---- 4. refit on everything and score customers as of today ----
    final_pipe = fit_calibrated(best_kind, snaps[features], snaps["Churn"].astype(int), snaps["CustomerID"])
    model_path = f"{model_dir}/best_churn_model.joblib"
    joblib.dump(final_pipe, model_path)

    today = transactions.build_snapshot(
        tx, tx["Date"].max(), active_days=active_days, horizon_days=window_days
    )
    today_prob = _positive(final_pipe, today[features]) if len(today) else test_prob
    thresholds = {
        "operating_threshold": threshold,
        "operating_threshold_rule": "balanced accuracy (Youden's J) on validation",
        "operating_threshold_selected_on": "validation",
        "operating_threshold_validation_precision": round(operating["precision"], 4),
        "operating_threshold_validation_recall": round(operating["recall"], 4),
        "operating_threshold_validation_f1": round(operating["f1"], 4),
        "medium_risk_threshold": round(float(np.quantile(today_prob, 0.65)), 4),
        "high_risk_threshold": round(float(np.quantile(today_prob, 0.85)), 4),
        "risk_band_population": "customers scored today (High = riskiest 15%, Medium = next 20%)",
    }
    with open(f"{report_dir}/decision_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    metadata = {
        "schema_version": 3,
        "training_mode": "temporal_transactions",
        "model_type": best_name,
        "feature_names": features,
        "row_count": int(len(snaps)),
        "window_days": window_days,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(f"{model_dir}/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    summary = {
        "source_label": source_label,
        "data_kind": "real_transactions",
        "window_days": window_days,
        "active_days": active_days,
        "n_cutoffs": len(cutoffs),
        "step_days": step_days,
        "split": split_info,
        "rows": {"train": len(train), "validation": len(val), "test": len(test),
                 "final_fit": int(len(snaps))},
        "selected_model": best_name,
        "operating_threshold": threshold,
        "operating_threshold_rule": thresholds["operating_threshold_rule"],
        "validation": comparison.drop(columns=["Key", "Selected"]).to_dict("records"),
        "test": {best_name: test_metrics},
        "test_previous_model": prev_metrics,
        "test_baseline_recency": base_metrics,
        "auc_gain_vs_previous_ci95": diff_prev_ci,
        "auc_gain_vs_baseline_ci95": diff_base_ci,
        "auc_gain_vs_previous": round(test_metrics["roc_auc"] - prev_metrics["roc_auc"], 4),
        "auc_gain_vs_baseline": round(test_metrics["roc_auc"] - base_metrics["roc_auc"], 4),
        "validation_threshold_metrics": {k: round(v, 4) for k, v in operating.items()},
        "features": features,
        "previous_features": legacy,
    }
    with open(f"{report_dir}/backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved model + backtest summary under {model_dir} and {report_dir}")
    return {
        "best_model_name": best_name,
        "best_model_path": model_path,
        "reused": False,
        "summary": summary,
    }


def _fit_previous(fit_set, legacy_features):
    """The model that shipped before this upgrade: old feature set, old forest settings."""
    forest = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=20,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )
    pipe = Pipeline([
        ("preprocessor", make_preprocessor("rf", legacy_features)),
        ("model", forest),
    ])
    return pipe.fit(fit_set[legacy_features], fit_set["Churn"].astype(int))
