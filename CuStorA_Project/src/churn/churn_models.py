# CuStorA - Churn Prediction Models
# Logistic Regression vs Random Forest vs MLP Neural Network

import os
import json
import sys
import hashlib
import warnings

import joblib
import pandas as pd
import sklearn

from sklearn.base import clone
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# 1. PATHS

DATA_PATH = "data/processed/customers_segmented.csv"

MODEL_DIR = "models/churn"
REPORT_DIR = "reports/churn"

# Columns not used for prediction: identifiers, dates, outputs from
# previous segmentation/analysis stages, or duplicates of a feature
# that's already included under a different name (e.g. Frequency IS
# TotalPurchases, CalculatedAOV IS AverageOrderValue). Feeding the
# model two copies of the same signal doesn't help accuracy, and it
# makes the SHAP explanations repeat the same reason twice under two
# different names -- so we keep one canonical copy of each.
COLUMNS_TO_DROP = [
    "CustomerID",
    "RegistrationDate",
    "LastPurchaseDate",
    # Previous segmentation outputs
    "Cluster",
    "SegmentName",
    "RFM_Segment",
    "RFM_Score",
    "RFM_Total",
    "R_Score",
    "F_Score",
    "M_Score",
    # Duplicates of a feature kept under another name
    "TotalPurchases",  # == Frequency
    "TotalSpent",  # == Monetary
    "TenureDays",  # == TenureYears, different unit
    "CalculatedAOV",  # == AverageOrderValue
    "EngagementLevel",  # binned version of EngagementScore
    "RecencyLevel",  # binned version of Recency
    # Near-zero correlation with Churn on the current dataset (see
    # reports/accuracy_improvement_log.md, iteration 1): |corr| < 0.02
    # for all four, versus 0.10-0.18 for the fields kept above. On the
    # shipped synthetic generator this isn't a fluke -- Age and
    # DiscountUsage are drawn independently of the hidden
    # relationship-quality trait that drives churn, and Recency/
    # TenureYears are derived from LastPurchaseDate/RegistrationDate,
    # which are themselves generated independently of that trait too
    # (see src/data/demo_generator.py's module docstring and churn_logit
    # formula -- neither date column is one of its inputs). Kept out of
    # the *model's* feature set only; still computed and shown elsewhere
    # in the app (Recency segments, tenure, etc. are useful presentation
    # features even where they don't help this particular label).
    "Age",
    "Recency",
    "TenureYears",
    "DiscountUsage",
    # Target
    "Churn",
]

# Candidate thresholds used to build the precision/recall-at-threshold
# table (see build_threshold_table). CuStorA's downstream 40%/60% risk
# bands (src/state/customer_state.py) are chosen from round numbers,
# not from this table -- this table is what makes that choice
# checkable, by showing what recall/precision/volume actually look
# like near those cut points.
THRESHOLD_CANDIDATES = [0.30, 0.40, 0.50, 0.60, 0.70]


def prepare_features(df):
    """Split a segmented dataframe into model features X and target y."""
    X = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")
    y = pd.to_numeric(df["Churn"], errors="coerce")
    valid_rows = y.notna()
    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].astype(int)
    return X, y


def build_preprocessor(X):
    """Build the ColumnTransformer used ahead of every candidate model.

    Includes pandas' nullable "string" dtype alongside "object"/"category"
    for categorical_features. This matters now that data can flow
    in-memory (see pipeline.py) rather than always round-tripping
    through a CSV: cleaning.py deliberately casts Gender to "string"
    dtype (data["Gender"].astype("string")), and previously, reading
    that column back from a CSV silently re-cast it to plain "object"
    dtype every time -- accidentally masking the fact that "string"
    wasn't in this include list. Feeding this exact in-memory DataFrame
    straight to this function (instead of a CSV-reloaded copy) would
    otherwise have silently dropped Gender from the model entirely.
    """
    categorical_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numerical_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32", "bool"]
    ).columns.tolist()
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
    return preprocessor, numerical_features, categorical_features


def get_candidate_models():
    """The three churn models CuStorA compares."""
    return {
        "Logistic Regression": LogisticRegression(
            # C and penalty tuned from the sklearn defaults (C=1.0,
            # penalty="l2") via a validation-set-only grid search (12
            # configs: C in {0.01,0.03,0.1,0.3,1,3} x {l1,l2}; test set
            # never touched -- see reports/accuracy_improvement_log.md,
            # iteration 4). Stronger L1 regularization (small C) both
            # improved validation ROC-AUC and, more importantly given
            # this is the model `select_best_model` actually picks,
            # raised validation Recall on the churn class from 0.50 to
            # 0.62 -- plausible given only ~600 training rows and ~20
            # candidate features, where a less-regularized model has
            # more room to fit noise instead of the (small) real signal.
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            C=0.03,
            penalty="l1",
            solver="liblinear",
        ),
        "Random Forest": RandomForestClassifier(
            # Depth/leaf-size constraints tightened from the previous
            # (n_estimators=300, max_depth=None, min_samples_split=5)
            # config -- see reports/accuracy_improvement_log.md,
            # iteration 3. An unconstrained forest was overfitting on
            # ~600 training rows: it memorized enough noise that its
            # predict_proba scores rarely crossed 0.5 for the minority
            # class (validation Recall was 0.06-0.00 across the last two
            # iterations). A small validation-set grid search over
            # depth/leaf-size (never touching the test set) found this
            # shallower, more regularized config improves both
            # validation ROC-AUC and Recall.
            n_estimators=100,
            max_depth=3,
            min_samples_split=20,
            min_samples_leaf=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            random_state=42,
        ),
    }


def save_confusion_matrix_chart(model_name, cm, report_dir):
    """Save a confusion matrix plot for one model."""
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Not Churn", "Churn"])
    plt.yticks([0, 1], ["Not Churn", "Churn"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(f"{report_dir}/{safe_name}_confusion_matrix.png", dpi=150)
    plt.close()


def train_and_evaluate(
    model_name, model, preprocessor, X_train, X_eval, y_train, y_eval, report_dir
):
    """Fit one model on the training data and score it on an evaluation
    set (the caller decides whether that's validation or test -- this
    run's model-comparison step passes validation, keeping the test
    set untouched until the final one-time evaluation)."""
    print("\n" + "=" * 60)
    print(f"Training: {model_name}")
    print("=" * 60)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_eval)
    y_probability = (
        pipeline.predict_proba(X_eval)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else pipeline.decision_function(X_eval)
    )
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_eval, y_pred),
        "Precision": precision_score(y_eval, y_pred, zero_division=0),
        "Recall": recall_score(y_eval, y_pred, zero_division=0),
        "F1": f1_score(y_eval, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_eval, y_probability),
        "Brier": brier_score_loss(y_eval, y_probability),
    }
    print(f"\nAccuracy : {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1 Score : {metrics['F1']:.4f}")
    print(f"ROC-AUC  : {metrics['ROC-AUC']:.4f}")
    print(f"Brier    : {metrics['Brier']:.4f}  (lower = predicted probabilities closer to reality)")
    print("\nClassification Report:")
    print(classification_report(y_eval, y_pred, zero_division=0))
    cm = confusion_matrix(y_eval, y_pred)
    print("Confusion Matrix:")
    print(cm)
    save_confusion_matrix_chart(model_name, cm, report_dir)
    return pipeline, metrics, y_probability


def select_best_model(results_df):
    """
    Selection metric: Recall first, F1 as a tiebreaker.

    CuStorA's stated goal is retention -- surfacing customers who are
    actually going to churn so the business can intervene. In that
    framing, a false negative (missing a real churner) is more costly
    than a false positive (flagging a customer who was fine, who then
    just gets an unnecessary outreach). That argues for optimizing
    Recall on the churn class over a metric like F1 or accuracy, which
    weight both error types equally.

    The tradeoff: prioritizing Recall alone can also pick a model that
    flags almost everyone as at-risk. F1 as a tiebreaker guards against
    that -- among models with similar Recall, it favors the one that
    isn't achieving that Recall by carpet-bombing predictions.
    """
    return results_df.sort_values(by=["Recall", "F1"], ascending=False)


def build_threshold_table(y_test, y_probability, thresholds=THRESHOLD_CANDIDATES):
    """
    Build a Recall / Precision / customers-flagged table across a set
    of candidate probability thresholds for the *best* churn model.

    This exists so the 40%/60% risk bands used downstream
    (src/state/customer_state.py) aren't a guess -- this table shows
    what actually happens to recall, precision, and outreach volume
    near those cut points, on this dataset.
    """
    rows = []
    y_probability = pd.Series(y_probability).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)
    for threshold in thresholds:
        predicted = (y_probability >= threshold).astype(int)
        flagged = int(predicted.sum())
        rows.append(
            {
                "Threshold": threshold,
                "Recall": recall_score(y_test, predicted, zero_division=0),
                "Precision": precision_score(y_test, predicted, zero_division=0),
                "CustomersFlagged": flagged,
            }
        )
    return pd.DataFrame(rows)


def save_precision_recall_curve(y_test, y_probability, output_path):
    """Save a precision-recall curve for the best model."""
    precision, recall, _ = precision_recall_curve(y_test, y_probability)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("CuStorA Churn Model: Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def calibrate_best_model(model_name, model, preprocessor, X_train, y_train):
    """
    Re-fit the winning model's classifier through CalibratedClassifierCV
    so its predicted probabilities are trustworthy, not just its
    ranking.

    ROC-AUC and Recall only depend on whether churners get *higher*
    scores than non-churners -- they say nothing about whether a
    customer scored at "70% churn probability" actually churns 70% of
    the time. CuStorA's earlier calibration table showed exactly that
    gap (e.g. a ~71% predicted bin observing only ~33% actual churn),
    which makes calling the raw model output a "probability"
    misleading, and makes any CLV or business-priority calculation
    downstream that multiplies by (1 - churn_probability) similarly
    unreliable.

    Implementation note: the preprocessor is fit once here and reused
    for both the calibrated classifier and prediction, and the
    calibrated estimator is wrapped back into a Pipeline with the same
    "preprocessor" / "model" step names as before -- so every
    downstream consumer of the saved model (customer_state.py's
    predict_proba call, shap_explainer.py's named_steps access) keeps
    working unmodified.
    """
    fitted_preprocessor = clone(preprocessor).fit(X_train)
    X_train_transformed = fitted_preprocessor.transform(X_train)

    # Isotonic regression is the more flexible / accurate calibration
    # method, but it needs enough data per fold to avoid overfitting
    # the calibration curve itself. Below ~1000 training rows, fall
    # back to Platt scaling (sigmoid), which assumes a simpler
    # logistic shape and is more stable on smaller samples.
    calibration_method = "isotonic" if len(X_train) >= 1000 else "sigmoid"
    calibrated_model = CalibratedClassifierCV(
        estimator=clone(model), method=calibration_method, cv=3
    )
    calibrated_model.fit(X_train_transformed, y_train)
    calibrated_pipeline = Pipeline(
        steps=[("preprocessor", fitted_preprocessor), ("model", calibrated_model)]
    )
    print(
        f"Calibrated {model_name} using {calibration_method} calibration "
        f"(cv=3, {len(X_train)} training rows)."
    )
    return calibrated_pipeline, calibration_method


def evaluate_temporal_generalization(df, model_name, model, preprocessor, report_dir):
    """
    Supplementary check: instead of a random train/test split, sort
    customers by LastPurchaseDate and train on the chronologically
    earlier customers, testing on the most recent ones. This is closer
    to the question a retention system actually has to answer --
    "can what I know today predict who churns next?" -- than a random
    split, which lets the model see examples from every point in time
    during training.

    Caveat: on the built-in synthetic dataset, LastPurchaseDate is
    drawn independently at random rather than reflecting a genuinely
    time-evolving customer base, so a temporal-vs-random performance
    gap here is not strong evidence either way. This check becomes
    meaningful once run against a real, time-ordered dataset -- it's
    included so that comparison is one command away, not a rebuild.
    """
    required = {"LastPurchaseDate", "Churn"}
    if not required.issubset(df.columns):
        return None
    dated = df.copy()
    dated["LastPurchaseDate"] = pd.to_datetime(dated["LastPurchaseDate"], errors="coerce")
    dated["Churn"] = pd.to_numeric(dated["Churn"], errors="coerce")
    dated = dated.dropna(subset=["LastPurchaseDate", "Churn"]).sort_values("LastPurchaseDate")
    if len(dated) < 100:
        return None
    X_all, y_all = prepare_features(dated)
    split_index = int(len(dated) * 0.8)
    X_train_t, X_test_t = X_all.iloc[:split_index], X_all.iloc[split_index:]
    y_train_t, y_test_t = y_all.iloc[:split_index], y_all.iloc[split_index:]

    # Not enough of both classes on one side of the split to score
    # meaningfully -- skip rather than report a misleading number.
    if y_train_t.nunique() < 2 or y_test_t.nunique() < 2 or len(X_test_t) < 20:
        print(
            "Skipping temporal validation: not enough churn/non-churn "
            "examples on one side of the chronological split."
        )
        return None
    temporal_pipeline = Pipeline(
        steps=[("preprocessor", clone(preprocessor)), ("model", clone(model))]
    )
    temporal_pipeline.fit(X_train_t, y_train_t)
    y_prob_t = temporal_pipeline.predict_proba(X_test_t)[:, 1]
    y_pred_t = temporal_pipeline.predict(X_test_t)
    metrics = {
        "Model": model_name,
        "SplitType": "Temporal (earliest 80% train / most recent 20% test)",
        "TrainRows": len(X_train_t),
        "TestRows": len(X_test_t),
        "ROC-AUC": roc_auc_score(y_test_t, y_prob_t),
        "Precision": precision_score(y_test_t, y_pred_t, zero_division=0),
        "Recall": recall_score(y_test_t, y_pred_t, zero_division=0),
        "F1": f1_score(y_test_t, y_pred_t, zero_division=0),
    }
    results_path = f"{report_dir}/temporal_validation.csv"
    pd.DataFrame([metrics]).to_csv(results_path, index=False)
    print(f"\nTemporal validation ({metrics['SplitType']}):")
    for key in ["ROC-AUC", "Precision", "Recall", "F1"]:
        print(f"  {key}: {metrics[key]:.4f}")
    print(f"Temporal validation results saved to:\n{results_path}")
    return metrics


def select_operating_threshold(y_test, y_probability, candidates=None):
    """
    Pick the single probability cutoff CuStorA actually uses to call a
    customer "predicted to churn" (customer_state.py's PredictedChurn
    flag), instead of a hardcoded 0.50.

    0.50 is the natural cutoff for a *symmetric* decision, but
    CuStorA's own stated goal (see select_best_model docstring) is
    asymmetric: missing a real churner is worse than a false alarm.
    It's also simply the wrong cutoff once probabilities are
    calibrated to a low base rate -- a calibrated model correctly
    outputs mostly-low probabilities when most customers don't churn,
    so a fixed 0.50 line can end up calling almost nobody "predicted
    to churn," silently collapsing Recall to ~0 even though the
    model's *ranking* of customers is still useful (see
    reports/churn/calibration_before_after.csv for why this matters).

    This scans a grid of candidate cutoffs and picks the one with the
    best F1 (precision/recall balance) on the test set, so the cutoff
    is chosen from what the data actually supports rather than a
    round number. Ties are broken by recall, in keeping with the
    retention framing.
    """
    if candidates is None:
        candidates = [round(t, 2) for t in [i / 100 for i in range(5, 96, 5)]]
    y_probability = pd.Series(y_probability).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)
    best = None
    for threshold in candidates:
        predicted = (y_probability >= threshold).astype(int)
        if predicted.sum() == 0:
            continue
        f1 = f1_score(y_test, predicted, zero_division=0)
        recall = recall_score(y_test, predicted, zero_division=0)
        precision = precision_score(y_test, predicted, zero_division=0)
        candidate_result = {
            "threshold": threshold,
            "f1": f1,
            "recall": recall,
            "precision": precision,
        }
        if best is None or (f1, recall) > (best["f1"], best["recall"]):
            best = candidate_result
    if best is None:
        best = {"threshold": 0.50, "f1": 0.0, "recall": 0.0, "precision": 0.0}
    return best


def save_decision_thresholds(operating_threshold_info, population_probability, report_dir):
    """
    Persist the cutoffs customer_state.py needs, derived from this run
    rather than hardcoded, as a small JSON config:

    - operating_threshold: the PredictedChurn yes/no cutoff (see
      select_operating_threshold), chosen on the VALIDATION set.
    - medium_risk_threshold / high_risk_threshold: cutoffs for the
      3-tier Low/Medium/High risk display, set at the 65th and 85th
      percentile of predicted churn probability across
      `population_probability`. Percentile-based bands mean "High
      risk" always means roughly "top 15% most at-risk in this
      dataset," which stays meaningful across datasets with very
      different overall churn rates -- unlike a fixed 60% cutoff,
      which means something different in a business with a 5% churn
      rate than one with a 40% churn rate.

    `population_probability` should be predicted probabilities for
    TRAIN + VALIDATION only, not the test set -- these percentile
    bands are a threshold decision like the operating threshold
    itself, so the test set stays out of computing them too, keeping
    it untouched until the one-time final evaluation.

    customer_state.py falls back to fixed defaults if this file is
    missing (e.g. in unit tests that don't run the full pipeline), so
    this is additive, not a hard dependency.
    """
    import numpy as np

    thresholds = {
        "operating_threshold": operating_threshold_info["threshold"],
        "operating_threshold_selected_on": "validation",
        "operating_threshold_validation_precision": round(operating_threshold_info["precision"], 4),
        "operating_threshold_validation_recall": round(operating_threshold_info["recall"], 4),
        "operating_threshold_validation_f1": round(operating_threshold_info["f1"], 4),
        "medium_risk_threshold": round(float(np.quantile(population_probability, 0.65)), 4),
        "high_risk_threshold": round(float(np.quantile(population_probability, 0.85)), 4),
        "risk_band_population": "train+validation (test set excluded)",
    }
    path = f"{report_dir}/decision_thresholds.json"
    with open(path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nDecision thresholds saved to:\n{path}")
    print(json.dumps(thresholds, indent=2))
    return thresholds


def build_calibration_table(y_test, y_probability, n_bins=5):
    """
    Answers "is a 73% churn probability actually a ~73% chance of
    churn?" -- bins predictions into probability ranges and compares
    the model's average predicted probability in each bin against the
    actual observed churn rate in that bin. A well-calibrated model
    has PredictedProbability roughly equal to ActualChurnRate in every
    row; a big gap means the number looks like a probability but
    isn't behaving like one yet.
    """
    y_test = pd.Series(y_test).reset_index(drop=True)
    y_probability = pd.Series(y_probability).reset_index(drop=True)
    actual_rate, predicted_rate = calibration_curve(
        y_test, y_probability, n_bins=n_bins, strategy="quantile"
    )
    bin_index = pd.qcut(y_probability, q=n_bins, duplicates="drop")
    counts = y_probability.groupby(bin_index, observed=True).size().to_numpy()
    table = pd.DataFrame(
        {
            "PredictedProbability": predicted_rate.round(3),
            "ActualChurnRate": actual_rate.round(3),
            "CustomerCount": counts[: len(predicted_rate)],
        }
    )
    table["CalibrationGap"] = (table["PredictedProbability"] - table["ActualChurnRate"]).round(3)
    return table


def save_calibration_plot(calibration_table, output_path):
    """Reliability diagram: predicted probability vs. observed churn
    rate per bin, against the perfect-calibration diagonal."""
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.plot(
        calibration_table["PredictedProbability"],
        calibration_table["ActualChurnRate"],
        marker="o",
        label="CuStorA best model",
    )
    plt.xlabel("Predicted churn probability")
    plt.ylabel("Actual churn rate")
    plt.title("Calibration: Is a Predicted Probability Trustworthy?")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def compute_data_fingerprint(X, y):
    """
    Content hash of the exact feature matrix + churn labels used to
    train a model -- not just their shape or column names. Two
    datasets can have identical schemas and still describe completely
    different customers, so a schema match alone is not enough to
    call a saved model reusable (see check_model_reusable). This
    fingerprint changes the moment a single feature value or label
    does, anywhere in the data.
    """
    x_hash = hashlib.sha256(pd.util.hash_pandas_object(X, index=True).values.tobytes()).hexdigest()
    y_hash = hashlib.sha256(pd.util.hash_pandas_object(y, index=True).values.tobytes()).hexdigest()
    return hashlib.sha256(f"{x_hash}:{y_hash}".encode()).hexdigest()


def build_model_metadata(best_model_name, X, calibration_method, data_fingerprint):
    """
    Everything check_model_reusable needs to verify a saved model
    against a *future* run's data, plus a few fields that are just
    useful to have on hand (trained_at, row_count) when looking at
    models/churn/model_metadata.json by eye.
    """
    return {
        "schema_version": 1,
        "model_type": best_model_name,
        "feature_names": list(X.columns),
        "feature_dtypes": {column: str(X[column].dtype) for column in X.columns},
        "row_count": int(len(X)),
        "data_fingerprint": data_fingerprint,
        "calibration_method": calibration_method,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }


def save_model_metadata(metadata, path):
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nModel metadata saved to:\n{path}")


def load_model_metadata(path):
    """Returns the saved metadata dict, or None if it's missing or
    unreadable (e.g. an older model saved before this metadata file
    existed) -- either way, the caller should treat that as
    "not reusable, retrain"."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def check_model_reusable(metadata, model_path, X, y):
    """
    Decide whether a previously saved churn model can be safely reused
    for THIS run's data instead of retraining, and explain why not if
    it can't.

    This is deliberately the strictest possible bar, not a fuzzy
    "close enough" check: reuse requires an exact match on
    data_fingerprint, i.e. the feature values and churn labels this
    model was trained on are provably, byte-for-byte, what this run is
    about to train on. That guarantees retraining would produce an
    identical model -- reusing it is then pure saved computation, with
    zero risk of silently applying a model fit on different customers
    or a different point in time to today's data. A schema match alone
    is NOT enough: two datasets can share the same column names and
    still describe a completely different customer base, which is
    exactly the "never assume a model trained on one dataset is
    universally valid for another" case this must guard against.

    Also requires the scikit-learn version to be unchanged, since
    unpickling a fitted estimator across a major sklearn version
    change can silently misbehave or fail outright.

    Returns (reusable: bool, reason: str).
    """
    if metadata is None:
        return False, "no saved model metadata found (first run, or an older model)"
    if not os.path.exists(model_path):
        return False, "model metadata exists but the model file is missing"
    saved_sklearn_version = metadata.get("sklearn_version")
    if saved_sklearn_version != sklearn.__version__:
        return False, (
            f"scikit-learn version changed ({saved_sklearn_version} -> {sklearn.__version__})"
        )
    if metadata.get("feature_names") != list(X.columns):
        return False, "feature schema (column names/order) changed since the saved model"
    current_fingerprint = compute_data_fingerprint(X, y)
    if metadata.get("data_fingerprint") != current_fingerprint:
        return False, "training data has changed since the saved model was trained"
    return True, "saved model's data fingerprint matches this run's data exactly"


def run(data=None, data_path=DATA_PATH, model_dir=MODEL_DIR, report_dir=REPORT_DIR, force_retrain=False):
    """
    Train and compare the three CuStorA churn models, select the best
    one by Recall-then-F1, and persist the model plus supporting
    evaluation artifacts (confusion matrices, PR curve, threshold
    table, model comparison table).

    `data=None` reads from `data_path`, same as before, for
    standalone/CLI use; pipeline.py instead passes the segmented
    DataFrame directly, in memory.

    Before doing any of that training work, checks whether a
    previously saved model (models/churn/best_churn_model.joblib +
    model_metadata.json) is safe to reuse for this exact data (see
    check_model_reusable) -- if so, loads it and returns immediately
    instead of retraining. `force_retrain=True` skips that check and
    always retrains (e.g. for an explicit "retrain now" action).
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    print("=" * 60)
    print("CuStorA - Churn Prediction")
    print("=" * 60)

    # NOTE ON THIS DATASET: if this run is using the built-in synthetic
    # generator (src/data/demo_generator.py) rather than a real uploaded
    # dataset, treat the metrics below as a demonstration of the pipeline
    # and evaluation methodology -- not evidence of real-world predictive
    # accuracy. Synthetic churn labels are generated from rules over the
    # same behavioural fields (engagement, recency, complaints, etc.) that
    # the model is trained on, so it's expected to score very well; a real
    # dataset, where churn arises from causes the model never sees, would
    # not perform this cleanly.

    if data is None:
        print("\nLoading dataset...")
        df = pd.read_csv(data_path)
    else:
        df = data
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nChurn distribution:")
    print(df["Churn"].value_counts())
    X, y = prepare_features(df)
    best_model_path = f"{model_dir}/best_churn_model.joblib"
    metadata_path = f"{model_dir}/model_metadata.json"

    # TRAIN / ANALYZE SEPARATION: reuse a compatible saved model
    # instead of retraining, if one exists (see check_model_reusable
    # for exactly what "compatible" requires).

    if not force_retrain:
        existing_metadata = load_model_metadata(metadata_path)
        reusable, reason = check_model_reusable(existing_metadata, best_model_path, X, y)
        if reusable:
            print(f"\nReusing existing churn model instead of retraining: {reason}")
            best_model = joblib.load(best_model_path)
            best_model_name = existing_metadata["model_type"]
            print(f"Loaded model: {best_model_name}  (trained {existing_metadata.get('trained_at')})")
            print(
                "Skipping training, calibration, and re-generation of the "
                "diagnostic reports in reports/churn/ -- they're still valid "
                "for this exact data and were not touched."
            )
            return {
                "best_model_name": best_model_name,
                "best_model_path": best_model_path,
                "reused": True,
                "reuse_reason": reason,
            }
        print(f"\nNot reusing a saved model ({reason}) -- training now.")

    preprocessor, numerical_features, categorical_features = build_preprocessor(X)
    print("\nNumerical features:")
    print(numerical_features)
    print("\nCategorical features:")
    print(categorical_features)

    # THREE-WAY SPLIT: TRAIN / VALIDATION / TEST

    # A single train/test split forces every decision -- which model
    # wins, which probability threshold to use -- to be made by
    # looking at the test set, and then that *same* test set gets
    # reused to report "final" metrics. That's test-set leakage: the
    # reported numbers are optimistic because the test set was part of
    # the decision-making process, not a held-out check on it.
    #
    # Instead:
    #   TRAIN      (60%) -- fit every candidate model
    #   VALIDATION (20%) -- compare candidates, calibrate, pick the
    #                        operating threshold
    #   TEST       (20%) -- touched exactly once, at the very end,
    #                        to report final metrics. Never used for
    #                        any decision above.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print("\nTraining samples  :", len(X_train))
    print("Validation samples:", len(X_val), " (model selection + threshold choice)")
    print("Test samples      :", len(X_test), " (untouched until final evaluation)")
    models = get_candidate_models()
    results = []
    trained_models = {}
    probabilities_by_model = {}
    for model_name, model in models.items():
        pipeline, metrics, y_probability = train_and_evaluate(
            model_name, model, preprocessor, X_train, X_val, y_train, y_val, report_dir
        )
        trained_models[model_name] = pipeline
        probabilities_by_model[model_name] = y_probability
        results.append(metrics)
    results_df = select_best_model(pd.DataFrame(results))
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (on VALIDATION set -- test set not used here)")
    print("=" * 60)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    results_path = f"{report_dir}/churn_model_comparison.csv"
    results_df.to_csv(results_path, index=False)
    print("\nModel comparison (validation-set metrics) saved to:")
    print(results_path)
    best_model_name = results_df.iloc[0]["Model"]
    uncalibrated_probability = probabilities_by_model[best_model_name]  # on validation
    print("\n" + "=" * 60)
    print("BEST CHURN MODEL")
    print("=" * 60)
    print(f"Selected model: {best_model_name}")

    # CALIBRATION: make the winning model's probabilities trustworthy
    # Re-fit the winning model type through CalibratedClassifierCV so
    # "PredictedChurnProbability" behaves like an actual probability
    # (see calibrate_best_model docstring). Everything downstream --
    # threshold table, calibration table/plot, the saved joblib model
    # that customer_state.py and shap_explainer.py load -- uses this
    # calibrated pipeline, not the raw one evaluated above.

    print("\nCalibrating best model's predicted probabilities...")
    raw_model = get_candidate_models()[best_model_name]
    best_model, calibration_method = calibrate_best_model(
        best_model_name, raw_model, preprocessor, X_train, y_train
    )
    # Threshold selection happens on VALIDATION, never on the test set --
    # this is the fix for the leakage the previous version had: it used
    # to pick the threshold on y_test and then report metrics computed
    # with that same threshold on that same y_test, which inflates the
    # reported numbers. best_probability below is on X_val.
    best_probability = best_model.predict_proba(X_val)[:, 1]

    # Pick the operating threshold now (before reporting Accuracy /
    # Precision / Recall / F1 below) so those numbers reflect the
    # cutoff CuStorA actually uses downstream, not the CalibratedClassifierCV
    # default of 0.50 -- which, at this dataset's ~15-20% churn rate,
    # calls almost nobody "predicted to churn" once probabilities are
    # properly calibrated (see select_operating_threshold docstring).
    operating_threshold_info = select_operating_threshold(y_val, best_probability)
    operating_threshold = operating_threshold_info["threshold"]
    best_prediction = (pd.Series(best_probability) >= operating_threshold).astype(int).to_numpy()
    calibrated_metrics = {
        "Model": f"{best_model_name} (calibrated, threshold={operating_threshold:.2f}, on VALIDATION)",
        "Accuracy": accuracy_score(y_val, best_prediction),
        "Precision": precision_score(y_val, best_prediction, zero_division=0),
        "Recall": recall_score(y_val, best_prediction, zero_division=0),
        "F1": f1_score(y_val, best_prediction, zero_division=0),
        "ROC-AUC": roc_auc_score(y_val, best_probability),
        "Brier": brier_score_loss(y_val, best_probability),
    }
    uncalibrated_brier = brier_score_loss(y_val, uncalibrated_probability)
    print(f"\nBrier score before calibration: {uncalibrated_brier:.4f}")
    print(
        f"Brier score after calibration : {calibrated_metrics['Brier']:.4f}  "
        f"(lower is better; this is the number that matters for "
        f"whether the probability itself, not just the ranking, "
        f"can be trusted)"
    )
    calibration_comparison_path = f"{report_dir}/calibration_before_after.csv"
    pd.DataFrame(
        [
            {
                "Model": best_model_name,
                "Stage": "Before calibration",
                "Brier": round(uncalibrated_brier, 4),
            },
            {
                "Model": best_model_name,
                "Stage": "After calibration",
                "Brier": round(calibrated_metrics["Brier"], 4),
            },
        ]
    ).to_csv(calibration_comparison_path, index=False)
    print(f"Before/after calibration comparison saved to:\n{calibration_comparison_path}")

    # Add the calibrated model's row to the saved model-comparison
    # table too, so the calibrated numbers actually used downstream
    # are visible next to the three candidates they were chosen from.
    results_with_calibrated = pd.concat(
        [results_df, pd.DataFrame([calibrated_metrics])], ignore_index=True
    )
    results_with_calibrated.to_csv(results_path, index=False)
    joblib.dump(best_model, best_model_path)
    print("\nBest (calibrated) model saved to:")
    print(best_model_path)

    # MODEL METADATA: what a *future* run needs to verify this saved
    # model is still safe to reuse (see check_model_reusable) instead
    # of retraining. Saved right alongside the model itself so the
    # two can never drift apart.
    data_fingerprint = compute_data_fingerprint(X, y)
    model_metadata = build_model_metadata(
        best_model_name, X, calibration_method, data_fingerprint
    )
    save_model_metadata(model_metadata, metadata_path)


    # TEMPORAL VALIDATION (supplementary check, see docstring)

    print("\nRunning temporal validation check...")
    evaluate_temporal_generalization(df, best_model_name, raw_model, preprocessor, report_dir)

    # PRECISION-RECALL CURVE + THRESHOLD TABLE (best model, on VALIDATION)
    # These are diagnostic / decision-support artifacts -- they're what
    # the operating threshold above was chosen from -- so, like that
    # threshold choice, they're built on validation, not on the test
    # set that's still untouched at this point.

    print("\nBuilding precision-recall curve and threshold table (on VALIDATION)...")
    pr_curve_path = f"{report_dir}/precision_recall_curve.png"
    save_precision_recall_curve(y_val, best_probability, pr_curve_path)
    threshold_table = build_threshold_table(y_val, best_probability)
    threshold_table_path = f"{report_dir}/threshold_table.csv"
    threshold_table.to_csv(threshold_table_path, index=False)
    print(threshold_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nThreshold table (validation-set) saved to:")
    print(threshold_table_path)

    # DECISION THRESHOLDS (replace hardcoded 0.50/0.40/0.60 cutoffs)
    # The operating threshold itself was already chosen above (used
    # for the calibrated Accuracy/Precision/Recall/F1 reported next to
    # the three candidate models); this section derives the remaining
    # Medium/High risk display bands and persists all three together.
    # See select_operating_threshold / save_decision_thresholds
    # docstrings: customer_state.py's Medium/High risk bands
    # (previously hardcoded at 40%/60%) had no connection to what this
    # model's probabilities actually look like.
    #
    # The population used to set the Medium/High percentile bands is
    # TRAIN + VALIDATION only -- deliberately excluding the test set,
    # since these percentile cut points are themselves a threshold
    # decision, and the test set stays out of every threshold decision
    # in this file, not just the primary operating threshold.

    print(
        f"\nOperating threshold (PredictedChurn cutoff): {operating_threshold:.2f}  "
        f"(validation Precision={operating_threshold_info['precision']:.3f}, "
        f"Recall={operating_threshold_info['recall']:.3f}, "
        f"F1={operating_threshold_info['f1']:.3f})"
    )
    train_and_val_probability = best_model.predict_proba(pd.concat([X_train, X_val], axis=0))[:, 1]
    thresholds_saved = save_decision_thresholds(
        operating_threshold_info, train_and_val_probability, report_dir
    )

    # CALIBRATION TABLE (is "73% churn probability" trustworthy?) -- VALIDATION

    print("\nChecking probability calibration for the calibrated model (on VALIDATION)...")
    calibration_table = build_calibration_table(y_val, best_probability)
    calibration_table_path = f"{report_dir}/calibration_table.csv"
    calibration_table.to_csv(calibration_table_path, index=False)
    calibration_plot_path = f"{report_dir}/calibration_plot.png"
    save_calibration_plot(calibration_table, calibration_plot_path)
    print(calibration_table.to_string(index=False))
    print("\nCalibration table (validation-set) saved to:")
    print(calibration_table_path)
    print("Calibration plot saved to:")
    print(calibration_plot_path)

    # FINAL TEST EVALUATION -- the test set's ONLY use in this run
    # Everything above -- which model won, calibration, the operating
    # threshold, the PR curve, the threshold table, the calibration
    # table, the risk-band percentiles -- was decided using only
    # TRAIN and VALIDATION. X_test / y_test have not been touched
    # anywhere above this point. This is the one time they're used:
    # a single, final scoring of the already-frozen model and
    # already-frozen threshold, computed once and not fed back into
    # any decision.

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION (test set touched once, here only)")
    print("=" * 60)
    test_probability = best_model.predict_proba(X_test)[:, 1]
    test_prediction = (pd.Series(test_probability) >= operating_threshold).astype(int).to_numpy()
    final_test_metrics = {
        "Model": f"{best_model_name} (calibrated, threshold={operating_threshold:.2f})",
        "Accuracy": accuracy_score(y_test, test_prediction),
        "Precision": precision_score(y_test, test_prediction, zero_division=0),
        "Recall": recall_score(y_test, test_prediction, zero_division=0),
        "F1": f1_score(y_test, test_prediction, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, test_probability),
        "Brier": brier_score_loss(y_test, test_probability),
        "TestRows": len(y_test),
    }
    print(f"Accuracy : {final_test_metrics['Accuracy']:.4f}")
    print(f"Precision: {final_test_metrics['Precision']:.4f}")
    print(f"Recall   : {final_test_metrics['Recall']:.4f}")
    print(f"F1 Score : {final_test_metrics['F1']:.4f}")
    print(f"ROC-AUC  : {final_test_metrics['ROC-AUC']:.4f}")
    print(f"Brier    : {final_test_metrics['Brier']:.4f}")
    test_cm = confusion_matrix(y_test, test_prediction)
    print("\nConfusion Matrix (final test evaluation):")
    print(test_cm)
    save_confusion_matrix_chart("final_test", test_cm, report_dir)
    final_test_path = f"{report_dir}/final_test_evaluation.csv"
    pd.DataFrame([final_test_metrics]).to_csv(final_test_path, index=False)
    print(f"\nFinal, one-time test evaluation saved to:\n{final_test_path}")
    print(
        "(This is the only file in reports/churn/ computed from the test "
        "set. Every other file above -- model comparison, threshold "
        "table, calibration table, decision thresholds -- was computed "
        "from train/validation only.)"
    )
    print("\n" + "=" * 60)
    print("CHURN MODEL TRAINING COMPLETE")
    print("=" * 60)
    return {
        "best_model_name": best_model_name,
        "best_model_path": best_model_path,
        "reused": False,
        "results_df": results_df,
        "threshold_table": threshold_table,
        "calibration_table": calibration_table,
        "final_test_metrics": final_test_metrics,
        "decision_thresholds": thresholds_saved,
    }


# COMMAND-LINE MODE


def main():
    run()


if __name__ == "__main__":
    main()
