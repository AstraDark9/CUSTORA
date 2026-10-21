# CuStorA - SHAP Explainability
# Explains why the churn model predicts higher or lower churn risk.

import os

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from src.churn import feature_contract

# 1. PATHS

DATA_PATH = "data/processed/customers_health.csv"
MODEL_PATH = "models/churn/best_churn_model.joblib"

REPORT_DIR = "reports/shap"

COLUMNS_TO_DROP = [
    "CustomerID",
    "RegistrationDate",
    "LastPurchaseDate",
    "Churn",
    "Cluster",
    "SegmentName",
    "RFM_Segment",
    "RelationshipHealthScore",
    "RelationshipHealth",
]

# Background set size used to build the explainer. SHAP only needs a
# representative sample of the training distribution to estimate a
# feature's baseline contribution -- passing the *entire* dataset as
# background (as this used to do) makes every explainer O(n) or worse
# per prediction explained, which is what made this stage blow up on
# anything beyond a few hundred rows. 100 rows is SHAP's own rule-of
# -thumb default background size and is plenty for a stable baseline.
BACKGROUND_SIZE = 100

# Global feature importance is computed over a capped number of rows
# (GLOBAL_IMPORTANCE_MAX_ROWS below) since a stable average needs a
# bigger sample than a single customer's top reasons do, but it's
# still capped rather than unbounded -- with the Tree/Linear explainers
# this cap barely matters (they're fast enough to run on the full
# dataset), but it keeps the MLP's Permutation fallback from becoming
# the bottleneck on very large datasets. (Detailed per-customer rows,
# when requested via compute_detailed_for_top_n below, are similarly
# capped by whatever N the caller passes.)
GLOBAL_IMPORTANCE_MAX_ROWS = 1000

# The Permutation explainer (MLP's fallback path, see build_explainer)
# is dramatically slower per row than Tree/Linear -- it has to run the
# model many times per row to estimate each feature's contribution
# numerically, instead of reading it off closed-form. 5000 rows is fine
# for Tree/Linear (they're fast regardless of row count) but can turn
# into a many-minutes-or-worse wait on the Permutation path, which is
# very likely what "the analysis never finishes" actually is if the
# MLP happened to win model comparison. Cap that path much lower.
PERMUTATION_GLOBAL_IMPORTANCE_MAX_ROWS = 200


def prepare_features(df, model=None):
    """Select the columns the churn model was trained on (see feature_contract)."""
    return feature_contract.select_model_features(df, COLUMNS_TO_DROP, model)


def _unwrap_base_estimator(model):
    """
    The saved "model" step is the winning estimator re-fit through
    CalibratedClassifierCV (see src/churn/churn_models.py), so
    type(model) is always CalibratedClassifierCV, never the Random
    Forest / Logistic Regression / MLP itself -- checking isinstance
    against CalibratedClassifierCV directly would always fall through
    to the slow generic path.

    Unwrap to one of the cross-validation fold's fitted base
    estimators so we can detect the real model type and use its fast
    SHAP path. Calibration only applies a monotonic rescaling to the
    output probabilities, it doesn't change which features drive the
    decision, so explaining the (fast, exact) base estimator instead
    of the (slow, wrapped) calibrated one is a reasonable trade for
    Tree/Linear models. We don't have that option for the MLP case
    below since there's no fast path regardless of calibration.
    """
    if isinstance(model, CalibratedClassifierCV):
        return model.calibrated_classifiers_[0].estimator
    return model


def build_explainer(model_pipeline, X):
    """
    Transform raw features through the model's own preprocessor, then
    build a SHAP explainer around the underlying fitted estimator
    (not just its predict_proba function) using a *sampled* background
    set instead of the full dataset.

    Passing the estimator object itself (rather than a bare predict_proba
    callable) lets SHAP recognize the model type and pick its fast,
    exact algorithm:
      - RandomForestClassifier / HistGradientBoostingClassifier -> TreeExplainer (exact, polynomial in
        tree depth, not dataset size)
      - LogisticRegression     -> LinearExplainer (closed-form, uses
        the model's coefficients directly)
      - anything else (here, MLPClassifier) -> no closed-form path
        exists, so we fall back to a sampled Permutation explainer
        over predict_proba, which is model-agnostic but must estimate
        contributions numerically and is the slow path.
    In every case the background is a small representative sample
    (shap.sample), not the full dataset, which is what actually fixes
    the O(n) explainer-construction cost.
    """
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]
    X_transformed = preprocessor.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_transformed = np.asarray(X_transformed)
    feature_names = preprocessor.get_feature_names_out()
    background_size = min(BACKGROUND_SIZE, len(X_transformed))
    background = shap.sample(X_transformed, background_size)
    base_estimator = _unwrap_base_estimator(model)
    if isinstance(base_estimator, (RandomForestClassifier, HistGradientBoostingClassifier)):
        explainer = shap.TreeExplainer(base_estimator, background)
    elif isinstance(base_estimator, LogisticRegression):
        explainer = shap.LinearExplainer(base_estimator, background)
    else:
        # MLPClassifier (or any other model without a closed-form
        # SHAP algorithm): fall back to a background-sampled,
        # model-agnostic explainer. Use the full calibrated pipeline's
        # predict_proba here (rather than the unwrapped base
        # estimator) since there's no speed advantage to unwrapping
        # on this path anyway, and it keeps the explained probability
        # scale consistent with what's actually shown to users.
        explainer = shap.PermutationExplainer(model.predict_proba, background)
    return explainer, X_transformed, feature_names


def compute_shap_values(explainer, X_transformed):
    """Run the SHAP explainer and normalize output shape to 2D (rows x features)."""
    shap_values = explainer.shap_values(X_transformed)
    shap_values = np.asarray(shap_values)

    # For binary classifiers, SHAP sometimes returns one array per
    # class (rows x features x classes) -- keep the "churn" class.
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    return shap_values


def build_global_importance(shap_values, feature_names):
    """Mean absolute SHAP value per feature, sorted descending."""
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({"Feature": feature_names, "MeanAbsoluteSHAP": mean_abs_shap})
    return importance_df.sort_values(by="MeanAbsoluteSHAP", ascending=False)


def build_customer_explanations(df, shap_values, feature_names, top_n=5):
    """Top N SHAP-driven reasons per customer, long-format."""
    customer_explanations = []
    for index in range(len(df)):
        customer_id = df.iloc[index]["CustomerID"]
        customer_shap = shap_values[index]
        top_indices = np.argsort(np.abs(customer_shap))[::-1][:top_n]
        for feature_index in top_indices:
            customer_explanations.append(
                {
                    "CustomerID": customer_id,
                    "Feature": feature_names[feature_index],
                    "SHAP_Value": customer_shap[feature_index],
                    "Impact": (
                        "Increases Churn Risk"
                        if customer_shap[feature_index] > 0
                        else "Decreases Churn Risk"
                    ),
                }
            )
    return pd.DataFrame(customer_explanations)


def explain_customer_rows(customer_rows_df, model_pipeline, explainer, feature_names, top_n=5):
    """
    On-demand SHAP explanation for one or a few already-selected
    customer rows, reusing an explainer already built for this model
    version (see build_explainer / app.py's _get_shap_explainer cache).

    This is the piece that replaced pre-computing detailed SHAP for up
    to a fixed number of customers on every single pipeline run (see
    run's docstring): a customer-profile view calls this
    the first time that customer's page is opened, then caches the
    (small) result, instead of looking it up in a batch file that may
    or may not have covered that customer. Uses the exact same
    preprocessing/feature-ordering/explainer as the batch path did --
    same model_pipeline.named_steps["preprocessor"], same explainer,
    same build_customer_explanations -- so the explanation is
    identical to what the old batch computation would have produced
    for this customer, just computed lazily instead of eagerly.
    """
    X = prepare_features(customer_rows_df, model_pipeline)
    X_transformed = model_pipeline.named_steps["preprocessor"].transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_transformed = np.asarray(X_transformed)
    shap_values = compute_shap_values(explainer, X_transformed)
    return build_customer_explanations(
        customer_rows_df.reset_index(drop=True), shap_values, feature_names, top_n=top_n
    )


def save_summary_plots(shap_values, X_transformed, feature_names, report_dir):
    """Save the SHAP beeswarm summary plot and the bar-chart summary plot."""
    plt.figure()
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{report_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    plt.figure()
    shap.summary_plot(
        shap_values, X_transformed, feature_names=feature_names, plot_type="bar", show=False
    )
    plt.tight_layout()
    plt.savefig(f"{report_dir}/shap_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()


def run(
    data=None,
    data_path=DATA_PATH,
    model=None,
    model_path=MODEL_PATH,
    report_dir=REPORT_DIR,
    save=True,
    compute_detailed_for_top_n=None,
):
    """
    Explain the trained churn model's predictions with SHAP: global
    feature importance + summary plots, always. Per-customer detailed
    explanations are NOT computed here by default any more (see Task 3
    / project review notes) -- compute_detailed_for_top_n=None skips
    them entirely, which is what pipeline.py now uses. Instead, a
    customer's detailed SHAP is computed on-demand the first time
    their profile page is opened, then cached -- see
    explain_customer_rows, called from app.py -- rather than
    pre-computed for up to a fixed number of customers on every single
    pipeline run whether or not anyone ever looks at them.

    Set compute_detailed_for_top_n=N to restore the old batch
    behavior (still available for standalone/CLI use, e.g. bulk
    exporting explanations for the N highest-risk customers) -- this
    still writes customer_shap_explanations.csv, same format as before.

    data=None/model=None load from disk (data_path/model_path) for
    standalone/CLI use; pipeline.py instead passes the health-scored
    DataFrame and the already trained-or-reused churn model pipeline
    directly, in memory.

    Two row caps keep the global side from scaling with dataset size
    (see BACKGROUND_SIZE / GLOBAL_IMPORTANCE_MAX_ROWS above): the
    explainer's background sample, and the sample global importance is
    averaged over -- both enough for a stable result without paying
    per-row cost for the whole dataset.
    """
    if save:
        os.makedirs(report_dir, exist_ok=True)
    print("=" * 60)
    print("CuStorA - SHAP Explainability")
    print("=" * 60)
    if data is None:
        print("\nLoading customer data...")
        data = pd.read_csv(data_path)
    df = data
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    if model is None:
        print("\nLoading best churn model...")
        model = joblib.load(model_path)
        print("Best churn model loaded successfully.")
    model_pipeline = model
    X = prepare_features(df, model_pipeline)
    print("\nTransforming features for SHAP...")
    explainer, X_transformed, feature_names = build_explainer(model_pipeline, X)
    print(f"Number of transformed features: {len(feature_names)}")
    print(f"Explainer: {type(explainer).__name__} (background size {BACKGROUND_SIZE})")
    n_rows = len(X_transformed)
    rng = np.random.RandomState(42)

    # GLOBAL IMPORTANCE (sampled if the dataset is large)
    # Use a much smaller cap on the slow Permutation path (MLP) than on
    # the fast Tree/Linear paths -- see PERMUTATION_GLOBAL_IMPORTANCE_MAX_ROWS
    # above for why.
    is_permutation = isinstance(explainer, shap.PermutationExplainer)
    row_cap = PERMUTATION_GLOBAL_IMPORTANCE_MAX_ROWS if is_permutation else GLOBAL_IMPORTANCE_MAX_ROWS
    global_sample_size = min(row_cap, n_rows)
    global_idx = rng.choice(n_rows, size=global_sample_size, replace=False)
    print(
        f"\nCalculating SHAP values for global importance "
        f"({global_sample_size} of {n_rows} customers)..."
    )
    global_shap_values = compute_shap_values(explainer, X_transformed[global_idx])
    importance_df = build_global_importance(global_shap_values, feature_names)
    importance_path = f"{report_dir}/global_feature_importance.csv"
    if save:
        importance_df.to_csv(importance_path, index=False)
        print("\nTop SHAP Features:")
        print(importance_df.head(15).to_string(index=False))
        print("\nCreating SHAP summary plots...")
        save_summary_plots(global_shap_values, X_transformed[global_idx], feature_names, report_dir)
    result = {"importance_df": importance_df}

    # PER-CUSTOMER EXPLANATIONS -- opt-in only, see docstring above.
    # The pipeline no longer requests this; a profile view computes
    # (and caches) one customer's explanation on demand instead, via
    # explain_customer_rows.

    if compute_detailed_for_top_n:
        detailed_n = min(compute_detailed_for_top_n, n_rows)
        if detailed_n < n_rows:
            churn_probability = model_pipeline.predict_proba(X)[:, 1]
            detailed_idx = np.argsort(churn_probability)[::-1][:detailed_n]
        else:
            detailed_idx = np.arange(n_rows)
        print(
            f"\nCalculating per-customer SHAP values for "
            f"{detailed_n} of {n_rows} customers (highest predicted churn risk)..."
        )
        detailed_shap_values = compute_shap_values(explainer, X_transformed[detailed_idx])
        detailed_df = df.iloc[detailed_idx].reset_index(drop=True)
        customer_explanation_df = build_customer_explanations(
            detailed_df, detailed_shap_values, feature_names
        )
        customer_explanation_path = f"{report_dir}/customer_shap_explanations.csv"
        if save:
            customer_explanation_df.to_csv(customer_explanation_path, index=False)
            print(f"\nPer-customer SHAP explanations saved to:\n{customer_explanation_path}")
        result["customer_explanation_df"] = customer_explanation_df

    print("\n" + "=" * 60)
    print("SHAP EXPLAINABILITY COMPLETE")
    print("=" * 60)
    if save:
        print("\nFiles created:")
        print(f"- {importance_path}")
        if compute_detailed_for_top_n:
            print(f"- {report_dir}/customer_shap_explanations.csv")
        print(f"- {report_dir}/shap_summary.png")
        print(f"- {report_dir}/shap_feature_importance.png")
    return result


# COMMAND-LINE MODE


def main():
    run()


if __name__ == "__main__":
    main()
