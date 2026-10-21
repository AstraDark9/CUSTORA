"""
CuStorA's full customer-intelligence pipeline.

Each stage below is an ordinary importable function from src/, called
directly in this process rather than as a separate subprocess -- see
project history for why (in-process is directly unit-testable, easier
to step through with a debugger, and avoids interpreter-startup
overhead per stage).

STAGE DATA FLOW (in-memory, checkpoints still written to disk):
Every stage module still writes the exact same on-disk checkpoint it
always has (data/processed/*.csv, reports/*, models/churn/*) -- that
part is unchanged, and matters for debugging, reproducibility, the
dashboard, customer profiles, and the viva ("show me what RFM produced
before clustering ran"). What changed is that this pipeline no longer
makes the NEXT stage re-read the file the PREVIOUS stage just wrote:
each stage module now exposes a `run(data=None, save=True)`-style
function that, given a DataFrame, returns the transformed DataFrame
directly -- this file holds that DataFrame in a local variable and
passes it straight to the next stage, in memory. Every stage's
`main()` (used for standalone/CLI runs) still reads from disk via
`run(data=None)`, so nothing about running a stage on its own changed.
This eliminates 11 of the pipeline's 12 on-disk reads (every stage
after the very first raw upload read) without eliminating a single
on-disk write.

MODEL REUSE: the churn stage checks whether a previously trained,
metadata-verified-compatible model already exists for this *exact*
data before retraining (see src/churn/churn_models.py's
check_model_reusable) -- retraining only happens when there's no
compatible model to reuse. See src/orchestration/run_manager.py's
CARRY_FORWARD_DIRS for how a previous run's model becomes a candidate
for this check in the first place.

SHAP: only global feature importance is computed here, once per run,
over a bounded sample of rows. Per-customer detailed explanations are
no longer batch-computed for every pipeline run -- they're computed
on-demand (and cached) the moment a customer's profile page is opened,
by app.py (see src/explainability/shap_explainer.py's
explain_customer_rows).
"""

import time
import traceback
from pathlib import Path

# Must come before importing any of the src.* modules below that
# `import matplotlib.pyplot` (kmeans, pca, churn_models,
# shap_explainer) -- matplotlib locks in its backend on the first
# pyplot import in the process. See the matching comment in app.py for
# why this has to be Agg specifically (every plot here is saved to a
# .png, never shown in a window; app.py additionally needs this
# because it runs the pipeline on a background thread, where a GUI
# backend crashes outright).
import matplotlib
matplotlib.use("Agg")

import joblib
import pandas as pd

from src.data import cleaning
from src.features import engineering
from src.rfm import rfm
from src.clustering import pca, kmeans, segment_naming
from src.churn import churn_models
from src.health import relationship_health
from src.explainability import shap_explainer
from src.state import customer_state
from src.recommendations import retention_recommendations
from src.business import clv_metrics
from src.orchestration import run_manager
from src.data import transactions
from src.churn import temporal_churn
import json

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

PIPELINE_STATUS_FILE = PROJECT_ROOT / "reports" / "pipeline_status.csv"

RAW_MESSY_PATH = PROJECT_ROOT / "data" / "raw" / "customers_messy.csv"

RUN_META_FILE = PROJECT_ROOT / "reports" / "run_meta.json"
STAGED_FILE = PROJECT_ROOT / "data" / "raw" / "uploads" / "staged.json"
COVERAGE_FILE = PROJECT_ROOT / "reports" / "data_coverage.csv"

# Model features computed from the ledger that engineering.py does not recompute are
# merged in generically (see _engineer below), so adding a feature to
# transactions.TEMPORAL_FEATURES needs no change here.


def load_staged():
    """What the user staged on the upload page (kind, mapping, labels), or {}."""
    if STAGED_FILE.exists():
        try:
            return json.loads(STAGED_FILE.read_text())
        except (OSError, ValueError):
            return {}
    return {}


def write_run_meta(meta):
    RUN_META_FILE.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta, created_at=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    RUN_META_FILE.write_text(json.dumps(meta, indent=2, default=str))


def read_ledger(path):
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


# ============================================================
# PIPELINE STAGES
# ============================================================
# (display name, callable). Every callable here takes the running
# DataFrame (or, for the two side-branch stages, nothing extra beyond
# what's already been computed) and returns the DataFrame the next
# stage should use -- see module docstring above.

PIPELINE_STAGES = [
    "Cleaning",
    "Feature Engineering",
    "RFM Analysis",
    "PCA",
    "K-Means Clustering",
    "Segment Naming",
    "Churn Prediction",
    "Relationship Health",
    "SHAP Explainability",
    "Customer State",
    "Retention Recommendations",
    "CLV and Business Metrics",
]


# ============================================================
# RUN ONE STAGE
# ============================================================


def run_stage(stage_name, stage_fn):
    """
    Run one stage and return (status record, stage_fn's return value).
    Raises RuntimeError on failure -- same behavior as before -- but
    the status record is built first so the caller can still write out
    what happened up to and including the failing stage.
    """
    print()
    print("=" * 70)
    print(f"RUNNING: {stage_name}")
    print("=" * 70)
    started_at = time.perf_counter()
    try:
        result = stage_fn()
    except Exception as error:
        elapsed = time.perf_counter() - started_at
        print(traceback.format_exc())
        wrapped = RuntimeError(f"{stage_name} failed: {error}")
        wrapped.status = {
            "Stage": stage_name,
            "Status": "FAILED",
            "Seconds": round(elapsed, 2),
            "Error": str(error),
        }
        raise wrapped from error
    elapsed = time.perf_counter() - started_at
    print(f"{stage_name} completed successfully.  ({elapsed:.2f}s)")
    status = {"Stage": stage_name, "Status": "OK", "Seconds": round(elapsed, 2), "Error": ""}
    return status, result


# ============================================================
# MAIN PIPELINE
# ============================================================


def run_pipeline(uploaded_dataset_path, stages=None, staged=None):
    """
    Run the full CuStorA pipeline against an uploaded dataset.

    `stages` defaults to the full PIPELINE_STAGES list (names only);
    tests or a future "re-run just clustering" feature can pass a
    subset instead. Passing a custom subset only changes which stage
    *statuses* get recorded -- every stage in the fixed data-flow below
    still runs, since later stages depend on earlier ones' in-memory
    output. This matches the previous behavior (`stages` was already
    display-only bookkeeping, not a real dependency-aware subset
    selector).
    """
    stage_names = stages if stages is not None else PIPELINE_STAGES
    staged = staged if staged is not None else load_staged()
    is_ledger = staged.get("kind") == "transactions"
    print()
    print("=" * 70)
    print("CuStorA - FULL CUSTOMER INTELLIGENCE PIPELINE")
    print("=" * 70)
    uploaded_dataset_path = Path(uploaded_dataset_path)

    # CHECK UPLOADED DATASET

    if not uploaded_dataset_path.exists():
        raise FileNotFoundError("Uploaded dataset was not found: " + str(uploaded_dataset_path))
    print()
    print("Uploaded dataset:")
    print(uploaded_dataset_path)

    # ISOLATE THIS RUN'S WORKSPACE
    # Every stage below still reads/writes the same fixed relative paths
    # it always has (data/processed/..., reports/..., models/churn/...).
    # start_new_run() atomically repoints those paths at a fresh
    # data/runs/<run_id>/ directory so this run's outputs can never be
    # clobbered by -- or clobber -- another run's, without changing any
    # stage module's own path handling. See src/orchestration/run_manager.py.

    ledger = {}
    if is_ledger:
        # Ledger mode: the uploaded file is a transaction history. The 13-column
        # customer table the later stages expect is built from it (and written
        # to data/raw so the run is inspectable), see src/data/transactions.py.
        # Building it is the first thing the "Cleaning" stage does below.
        seed_path = uploaded_dataset_path.parent / "customers_from_transactions.csv"
        pd.DataFrame(columns=transactions.CANONICAL_COLUMNS).to_csv(seed_path, index=False)
        run_id = run_manager.start_new_run(seed_path)
    else:
        run_id = run_manager.start_new_run(uploaded_dataset_path)
    print()
    print("Run ID:")
    print(run_id)
    print("(isolated under data/runs/" + run_id + "/)")

    # RUN ALL STAGES, THREADING THE DATAFRAME BETWEEN THEM IN MEMORY
    # (see module docstring for what this does and doesn't change)

    completed_stages = []
    stage_statuses = []
    try:
        if is_ledger:

            def _prepare_and_clean():
                raw_ledger = read_ledger(uploaded_dataset_path)
                tx, tx_report = transactions.prepare_transactions(raw_ledger, staged["mapping"])
                window_days = int(staged.get("window_days", 90))
                active_days = int(staged.get("active_days", window_days))
                as_of = tx["Date"].max()
                everyone = transactions.build_snapshot(tx, as_of)
                snap = everyone.loc[everyone["Recency"] <= active_days].reset_index(drop=True)
                if len(snap) < 30:
                    raise ValueError(
                        f"Only {len(snap)} customers bought in the last {active_days} days "
                        "before the end of the file - too few to analyse."
                    )
                canonical = transactions.to_canonical_table(snap)
                canonical.to_csv(RAW_MESSY_PATH, index=False)
                ledger.update(
                    tx=tx, report=tx_report, as_of=as_of, snap=snap,
                    window_days=window_days, active_days=active_days,
                    customers_total=len(everyone),
                )
                coverage = transactions.coverage_report(tx_report, tx)
                COVERAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
                coverage.to_csv(COVERAGE_FILE, index=False)
                return cleaning.run(canonical)

            status, cleaned_df = run_stage("Cleaning", _prepare_and_clean)
        else:
            raw_df = pd.read_csv(RAW_MESSY_PATH)
            status, cleaned_df = run_stage("Cleaning", lambda: cleaning.run(raw_df))
        stage_statuses.append(status)
        completed_stages.append("Cleaning")
        save_pipeline_status(stage_statuses, stage_names)

        def _engineer():
            if not is_ledger:
                return engineering.run(cleaned_df)
            out = engineering.run(cleaned_df, reference_date=ledger["as_of"])
            missing = [f for f in transactions.TEMPORAL_FEATURES if f not in out.columns]
            extra = ledger["snap"][["CustomerID"] + missing].copy()
            extra["CustomerID"] = extra["CustomerID"].astype(str)
            out["CustomerID"] = out["CustomerID"].astype(str)
            return out.merge(extra, on="CustomerID", how="left")

        status, features_df = run_stage("Feature Engineering", _engineer)
        stage_statuses.append(status)
        completed_stages.append("Feature Engineering")
        save_pipeline_status(stage_statuses, stage_names)

        status, rfm_df = run_stage("RFM Analysis", lambda: rfm.run(features_df))
        stage_statuses.append(status)
        completed_stages.append("RFM Analysis")
        save_pipeline_status(stage_statuses, stage_names)

        # PCA is a visualization-only side branch (see pca.py's
        # add_pca_columns docstring): K-Means clusters on the RFM
        # features directly, not on PC1/PC2, so its output isn't
        # needed by any later stage -- still run and checkpointed for
        # the dashboard's PCA scatter plot.
        status, _pca_df = run_stage("PCA", lambda: pca.run(rfm_df))
        stage_statuses.append(status)
        completed_stages.append("PCA")
        save_pipeline_status(stage_statuses, stage_names)

        status, clustered_df = run_stage("K-Means Clustering", lambda: kmeans.run(rfm_df))
        stage_statuses.append(status)
        completed_stages.append("K-Means Clustering")
        save_pipeline_status(stage_statuses, stage_names)

        status, segmented_df = run_stage(
            "Segment Naming", lambda: segment_naming.run(clustered_df)
        )
        stage_statuses.append(status)
        completed_stages.append("Segment Naming")
        save_pipeline_status(stage_statuses, stage_names)

        # Churn Prediction: trains a model, or reuses a compatible one
        # already trained on this exact data (see churn_models.py's
        # check_model_reusable) -- either way, doesn't modify the
        # dataframe itself. Load the (trained-or-reused) model object
        # once here so downstream stages don't each load it from disk
        # again.
        if is_ledger:
            status, churn_result = run_stage(
                "Churn Prediction",
                lambda: temporal_churn.run(
                    ledger["tx"],
                    window_days=ledger["window_days"],
                    active_days=ledger["active_days"],
                    source_label=staged.get("label", "Uploaded transactions"),
                ),
            )
        else:
            status, churn_result = run_stage(
                "Churn Prediction", lambda: churn_models.run(segmented_df)
            )
        stage_statuses.append(status)
        completed_stages.append("Churn Prediction")
        save_pipeline_status(stage_statuses, stage_names)
        churn_model = joblib.load(churn_result["best_model_path"])

        status, health_df = run_stage(
            "Relationship Health", lambda: relationship_health.run(segmented_df)
        )
        stage_statuses.append(status)
        completed_stages.append("Relationship Health")
        save_pipeline_status(stage_statuses, stage_names)

        # SHAP: global feature importance only (see module docstring).
        status, _shap_result = run_stage(
            "SHAP Explainability", lambda: shap_explainer.run(data=health_df, model=churn_model)
        )
        stage_statuses.append(status)
        completed_stages.append("SHAP Explainability")
        save_pipeline_status(stage_statuses, stage_names)

        status, state_df = run_stage(
            "Customer State", lambda: customer_state.run(data=health_df, model=churn_model)
        )
        stage_statuses.append(status)
        completed_stages.append("Customer State")
        save_pipeline_status(stage_statuses, stage_names)

        status, recommendations_df = run_stage(
            "Retention Recommendations", lambda: retention_recommendations.run(state_df)
        )
        stage_statuses.append(status)
        completed_stages.append("Retention Recommendations")
        save_pipeline_status(stage_statuses, stage_names)

        status, _business_df = run_stage(
            "CLV and Business Metrics",
            lambda: clv_metrics.run(recommendations_df, window_days=ledger.get("window_days", 90)),
        )
        stage_statuses.append(status)
        completed_stages.append("CLV and Business Metrics")
        save_pipeline_status(stage_statuses, stage_names)
    except RuntimeError as error:
        # Stage failed. Record its status (attached by run_stage),
        # write out what we know so far, then re-raise -- callers
        # (app.py, main()) still see the same failure as before.
        if hasattr(error, "status"):
            stage_statuses.append(error.status)
        save_pipeline_status(stage_statuses, stage_names)
        print()
        print("Last successful stage:", completed_stages[-1] if completed_stages else "None")
        raise

    if is_ledger:
        write_run_meta(
            {
                "data_kind": "real_transactions",
                "source_label": staged.get("label", "Uploaded transactions"),
                "currency": staged.get("currency", "\u00a3"),
                "as_of": ledger["as_of"].strftime("%Y-%m-%d"),
                "window_days": ledger["window_days"],
                "active_days": ledger["active_days"],
                "customers_scored": int(len(ledger["snap"])),
                "customers_total": int(ledger["customers_total"]),
                "ledger": ledger["report"],
                "model": churn_result.get("summary", {}).get("selected_model"),
                "test_roc_auc": churn_result.get("summary", {})
                .get("test", {})
                .get(churn_result.get("best_model_name"), {})
                .get("roc_auc"),
            }
        )
    else:
        COVERAGE_FILE.unlink(missing_ok=True)
        write_run_meta(
            {
                "data_kind": staged.get("data_kind", "customer_table"),
                "source_label": staged.get("label", "Uploaded customer table"),
                "currency": staged.get("currency", "\u20b9"),
                "customers_scored": int(len(state_df)),
                "customers_total": int(len(state_df)),
            }
        )

    # FINAL CHECK

    final_output = PROJECT_ROOT / "data" / "processed" / "customers_business.csv"
    if not final_output.exists():
        save_pipeline_status(stage_statuses, stage_names)
        raise FileNotFoundError("Pipeline finished but the final business dataset was not created.")
    print()
    print("=" * 70)
    print("CUSTORA PIPELINE COMPLETE")
    print("=" * 70)
    print()
    print("Completed stages:")
    for stage in completed_stages:
        print("\u2713", stage)
    save_pipeline_status(stage_statuses, stage_names)
    print()
    print("Final dataset:")
    print(final_output)
    return run_id


def save_pipeline_status(stage_statuses, all_stages):
    """
    Save a per-stage status table (status, seconds, error) so a run
    can be debugged or presented without scrolling back through the
    console log. Stages that never got a chance to run (because an
    earlier stage failed) are recorded as SKIPPED.
    """
    ran_names = {row["Stage"] for row in stage_statuses}
    rows = list(stage_statuses)
    for stage_name in all_stages:
        if stage_name not in ran_names:
            rows.append({"Stage": stage_name, "Status": "SKIPPED", "Seconds": "", "Error": ""})
    status_df = pd.DataFrame(rows)
    PIPELINE_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    status_df.to_csv(PIPELINE_STATUS_FILE, index=False)
    print()
    print("Pipeline stage status:")
    icons = {"OK": "\u2713", "FAILED": "\u2717", "SKIPPED": "-"}
    for row in rows:
        icon = icons.get(row["Status"], "?")
        seconds = f"{row['Seconds']}s" if row["Seconds"] != "" else ""
        print(f"{icon} {row['Stage']:<28} {row['Status']:<8} {seconds}")
        if row.get("Error"):
            print(f"    error: {row['Error']}")
    print()
    print("Full status saved to:", PIPELINE_STATUS_FILE)


# ============================================================
# COMMAND-LINE MODE
# ============================================================


def main():
    """
    Command-line entry point. If a dataset was already uploaded through
    the web app, use that. Otherwise ask how many customers to generate
    and create a synthetic dataset to run the pipeline on.
    """
    uploaded_dataset = PROJECT_ROOT / "data" / "raw" / "uploads" / "customers_uploaded.csv"
    if uploaded_dataset.exists():
        run_pipeline(uploaded_dataset)
        return
    from src.data.demo_generator import save_demo_dataset

    print("No dataset found.")
    count = input("How many customers should I generate? [1000]: ").strip()
    count = int(count) if count.isdigit() else 1000
    generated_path = save_demo_dataset(count, uploaded_dataset)
    run_pipeline(generated_path)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
