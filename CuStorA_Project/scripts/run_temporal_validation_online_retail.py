"""
Trains and evaluates churn models on the real, temporally-honest
snapshot dataset built by build_temporal_dataset.py.

Implements roadmap items:
  #16 time-based train/validation/test split (no random shuffling across time)
  #18 stronger models: XGBoost, LightGBM, alongside the existing
      LogReg / RandomForest / MLP for a fair comparison
  #20 light hyperparameter search on the validation set only
  #21 evaluate ROC-AUC, PR-AUC, F1, Brier -- not just recall
  #22 Precision@K / Recall@K
  #23 Lift@K
  #25 class-imbalance handling via class_weight / scale_pos_weight
  #26 a simple weighted ensemble of the two best models
  #28 probability calibration (isotonic) on top of the best model
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, brier_score_loss,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

DATA_PATH = "data/processed/online_retail_temporal_snapshots.csv"
REPORT_PATH = "reports/external_validation_temporal_online_retail/model_comparison.json"

TRAIN_CUTOFFS = ["2011-03-31", "2011-04-30", "2011-05-31", "2011-06-30"]
VAL_CUTOFFS = ["2011-07-31"]
TEST_CUTOFFS = ["2011-08-31"]

FEATURES = [
    "Recency", "TenureDays", "Frequency", "Monetary", "AverageOrderValue",
    "UniqueProducts", "AvgBasketSize",
    "Orders_30d", "Orders_Prev30d", "Spend_30d", "Spend_Prev30d",
    "Orders_90d", "Orders_Prev90d", "Spend_90d", "Spend_Prev90d",
    "FreqTrendRatio", "SpendTrendRatio", "SpendChange30d", "FreqChange90d",
    "ActivitySlope", "ActivityVolatility", "SpendSlope", "SpendVolatility",
    "RecencyXMonetary", "RecencyOverTenure",
    "R_score", "F_score", "M_score", "RFM_Score",
    "Country",
]
CATEGORICAL = ["Country"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]
TARGET = "Churn"


def load_splits():
    df = pd.read_csv(DATA_PATH)
    train = df[df["cutoff_date"].isin(TRAIN_CUTOFFS)]
    val = df[df["cutoff_date"].isin(VAL_CUTOFFS)]
    test = df[df["cutoff_date"].isin(TEST_CUTOFFS)]
    return train, val, test


def make_preprocessor():
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", cat_pipe, CATEGORICAL),
    ])


def business_metrics(y_true, y_prob, k_fracs=(0.10, 0.20)):
    n = len(y_true)
    base_rate = np.mean(y_true)
    order = np.argsort(-y_prob)
    y_sorted = np.asarray(y_true)[order]
    out = {}
    for k in k_fracs:
        top_n = max(1, int(round(n * k)))
        top = y_sorted[:top_n]
        precision_k = top.mean()
        recall_k = top.sum() / max(y_sorted.sum(), 1)
        lift_k = precision_k / base_rate if base_rate > 0 else float("nan")
        out[f"precision_at_{int(k*100)}pct"] = round(float(precision_k), 4)
        out[f"recall_at_{int(k*100)}pct"] = round(float(recall_k), 4)
        out[f"lift_at_{int(k*100)}pct"] = round(float(lift_k), 4)
    return out


def evaluate(name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "precision_at_0.5": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_at_0.5": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_at_0.5": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
    }
    metrics.update(business_metrics(y_true, y_prob))
    print(f"  [{name}] ROC-AUC={metrics['roc_auc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}  "
          f"Lift@10%={metrics['lift_at_10pct']:.2f}x  Brier={metrics['brier_score']:.3f}")
    return metrics


def build_models(pos_weight):
    return {
        "LogisticRegression": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)),
        ]),
        "RandomForest": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=8, min_samples_leaf=10,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
        ]),
        "MLP": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                                   early_stopping=True, random_state=42)),
        ]),
        "XGBoost": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", xgb.XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5, reg_lambda=1.0,
                scale_pos_weight=pos_weight, eval_metric="logloss",
                random_state=42, n_jobs=-1)),
        ]),
        "LightGBM": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", lgb.LGBMClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20, reg_lambda=1.0,
                scale_pos_weight=pos_weight, random_state=42,
                n_jobs=-1, verbosity=-1)),
        ]),
    }


def main():
    train, val, test = load_splits()
    print(f"Train: {len(train):,} rows (cutoffs {TRAIN_CUTOFFS})")
    print(f"Val:   {len(val):,} rows (cutoff {VAL_CUTOFFS})")
    print(f"Test:  {len(test):,} rows (cutoff {TEST_CUTOFFS}) -- held out, touched once\n")
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models = build_models(pos_weight)
    results = {}
    fitted = {}
    print("--- Validation performance (model selection) ---")
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        val_prob = pipe.predict_proba(X_val)[:, 1]
        results[name] = {"validation": evaluate(name, y_val, val_prob)}
        fitted[name] = pipe
    best_name = max(results, key=lambda n: results[n]["validation"]["roc_auc"])
    print(f"\nBest single model on validation (by ROC-AUC): {best_name}")

    # simple weighted ensemble of the top-2 validation performers
    ranked = sorted(results, key=lambda n: results[n]["validation"]["roc_auc"], reverse=True)
    top2 = ranked[:2]
    val_probs = {n: fitted[n].predict_proba(X_val)[:, 1] for n in top2}
    best_w, best_auc = 0.5, -1
    for w in np.arange(0.0, 1.01, 0.1):
        blend = w * val_probs[top2[0]] + (1 - w) * val_probs[top2[1]]
        auc = roc_auc_score(y_val, blend)
        if auc > best_auc:
            best_auc, best_w = auc, w
    print(f"Ensemble of {top2}: weight {best_w:.1f}/{1-best_w:.1f} -> val ROC-AUC {best_auc:.4f}")
    print("\n--- Held-out TEST performance (touched once, after model selection) ---")
    for name, pipe in fitted.items():
        test_prob = pipe.predict_proba(X_test)[:, 1]
        results[name]["test"] = evaluate(name, y_test, test_prob)
    test_probs = {n: fitted[n].predict_proba(X_test)[:, 1] for n in top2}
    ensemble_test_prob = best_w * test_probs[top2[0]] + (1 - best_w) * test_probs[top2[1]]
    results["Ensemble_" + "_".join(top2)] = {
        "weight": {top2[0]: best_w, top2[1]: round(1 - best_w, 2)},
        "validation": {"roc_auc": round(best_auc, 4)},
        "test": evaluate("Ensemble", y_test, ensemble_test_prob),
    }

    # calibrate the best single model with isotonic regression, fit on
    # validation only, evaluated on the held-out test set
    best_pipe = fitted[best_name]
    if FrozenEstimator is not None:
        calibrated = CalibratedClassifierCV(FrozenEstimator(best_pipe), method="isotonic")
    else:
        calibrated = CalibratedClassifierCV(best_pipe, method="isotonic", cv="prefit")
    calibrated.fit(X_val, y_val)
    calib_test_prob = calibrated.predict_proba(X_test)[:, 1]
    results[f"{best_name}_calibrated"] = {
        "test": evaluate(f"{best_name}+isotonic", y_test, calib_test_prob)
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull comparison written to {REPORT_PATH}")
    print("\n=== SUMMARY (held-out test set, chronologically after training data) ===")
    for name, r in results.items():
        if "test" in r:
            t = r["test"]
            print(f"{name:28s} ROC-AUC={t['roc_auc']:.3f}  PR-AUC={t['pr_auc']:.3f}  "
                  f"Lift@10%={t['lift_at_10pct']:.2f}x  Precision@10%={t['precision_at_10pct']:.2f}  "
                  f"Brier={t['brier_score']:.3f}")


if __name__ == "__main__":
    main()
