import os
import sys
import threading

# Force stdout to line-buffer (flush after every print) instead of
# fully buffering. Fully-buffered stdout is common when stdout isn't a
# real terminal (VS Code's Python debugger, some Windows terminals,
# `python app.py` piped anywhere) -- without this, every "RUNNING:
# <stage>" / "<stage> completed (Xs)" print from the background
# pipeline thread can sit invisible in a buffer until the process
# exits, making a perfectly normal multi-minute run look completely
# frozen in the console even though it's progressing fine.
sys.stdout.reconfigure(line_buffering=True)

# Force matplotlib's non-interactive Agg backend, before anything below
# (pipeline.py -> src/clustering/kmeans.py, pca.py, churn_models.py,
# shap_explainer.py, etc.) gets a chance to `import matplotlib.pyplot`
# and let matplotlib pick its own default GUI backend instead (TkAgg on
# a typical Windows install, since Tk ships with Python there).
#
# That matters because the pipeline runs on a background thread (see
# _run_pipeline_in_background below), and every GUI backend -- Tk
# included -- only works from the main thread. Every plot the pipeline
# draws was hitting that from the wrong thread: "RuntimeError: main
# thread is not in main loop" on each figure/widget as it got garbage
# collected, and eventually "Tcl_AsyncDelete: async handler deleted by
# the wrong thread", which is fatal to the whole Tcl interpreter and
# kills the process outright -- not just the pipeline run.
#
# Agg is the right backend here regardless of threading: every plot in
# this app is saved straight to a .png (reports/pca_visualization.png,
# reports/churn/*.png, reports/shap/*.png) and never shown in a window,
# which is exactly what Agg is for. Must run before the first
# `import matplotlib.pyplot` anywhere in the process -- matplotlib
# locks in its backend on that first import, so setting this later
# (even earlier in this same file, just after a pyplot import had
# already happened) would be too late.
import matplotlib
matplotlib.use("Agg")

import joblib
from datetime import datetime
from pathlib import Path

import pandas as pd
import json

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    abort,
    jsonify,
    send_from_directory,
)

# CuStorA full analysis pipeline
from pipeline import (
    run_pipeline,
    PIPELINE_STAGES,
    save_pipeline_status,
    STAGED_FILE,
    RUN_META_FILE,
    COVERAGE_FILE,
    load_staged,
)
from src.data import transactions
from src.explainability import reason_translator
from src.data.demo_generator import save_demo_dataset
from src.data import column_mapper
from src.explainability import shap_explainer
from src.explainability.reason_translator import build_customer_reasons, base_feature_name
from src.outreach import reached_store
from src.orchestration import stage_report

app = Flask(__name__)

# ==================================================
# DEBUG MODE
# ==================================================

# Debug mode (and with it, the Werkzeug interactive debugger) must
# never be on outside a trusted local machine -- the debugger allows
# arbitrary code execution from the browser if it's ever reachable
# over a network. It is OFF by default and only turns on when
# FLASK_DEBUG=1 is explicitly set in the environment.
FLASK_DEBUG = os.environ.get("FLASK_DEBUG") == "1"

# ==================================================
# SECRET KEY
# ==================================================

# Secret key is required for flash messages and CSRF tokens. It must
# come from the environment so it isn't checked into source control
# or shared between deployments. In debug mode we fall back to a
# clearly-labeled dev-only value so `flask run` still works locally
# without extra setup; outside debug mode a missing SECRET_KEY is a
# hard startup failure.
# NOTE (temporary, local-dev only): login is disabled for now (see the
# AUTHENTICATION section below), so this always falls back to a
# dev-only key instead of hard-failing when SECRET_KEY isn't set.
# Before deploying this anywhere reachable by anyone else, set a real
# SECRET_KEY env var and re-enable login.
_secret_key = os.environ.get("SECRET_KEY") or "insecure-local-dev-key-do-not-use-in-production"

app.secret_key = _secret_key

# ==================================================
# UPLOAD SIZE LIMIT
# ==================================================

# Reject request bodies over 20 MB before they're fully read into
# memory, so a large/adversarial upload can't be used to exhaust
# server memory or disk. Flask turns this into a 413 response, which
# is handled below with a friendly flash message on the upload page.
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB (transaction files are large)

# ==================================================
# AUTHENTICATION -- DISABLED (temporary)
# ==================================================
# Login (Flask-Login) and CSRF protection (Flask-WTF) are turned off
# for now so the app runs with just `pip install -r requirements.txt`
# and no extra env vars, and so nobody has to fight version-pin issues
# in those two packages to get a local demo running.
#
# Every route below is still decorated with @login_required -- rather
# than strip that off each one (and have to remember to add it back),
# it's a no-op here so every page is open. `csrf_token()` is also
# faked out below since a couple of templates still call it in a
# hidden form field.
#
# To bring real login back later: add Flask-Login/Flask-WTF back to
# requirements.txt, restore the AUTH_USERNAME/AUTH_PASSWORD/DemoUser/
# LoginManager setup and the real login_required import, and delete
# this block.


def login_required(view_func):
    return view_func


@app.context_processor
def _inject_dummy_csrf_token():
    return {"csrf_token": lambda: ""}


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


@app.context_processor
def _inject_run_meta():
    """What the current analysis is based on (real ledger / synthetic / uploaded table),
    available to every template as `run_meta`, plus the currency symbol to show."""
    meta = _read_json(RUN_META_FILE) or {}
    currency = meta.get("currency") or "\u20b9"
    reason_translator.CURRENCY_SYMBOL = currency
    return {"run_meta": meta, "currency": currency}


# ==================================================
# REACHED CUSTOMERS
# ==================================================
# Customers your team has already contacted. They leave the action queue, the
# matrix and the planner (so nobody is chased twice) and are listed on the
# "Reached customers" page instead. See src/outreach/reached_store.py.


def _dataset_fingerprint():
    """Identifies the dataset behind the current analysis (the reached list belongs to one dataset)."""
    meta = _read_json(RUN_META_FILE) or {}
    return f"{meta.get('source_label', '')}|{meta.get('as_of', '')}|{meta.get('customers_total', '')}"


def _reached_records():
    return reached_store.load(_dataset_fingerprint())


@app.context_processor
def _inject_reached_count():
    try:
        count = len(_reached_records())
    except Exception:  # the sidebar badge must never break a page
        count = 0
    return {"reached_count": count}


# ==================================================
# DATA PATHS
# ==================================================

BUSINESS_DATA_PATH = "data/processed/customers_business.csv"

CHURN_MODEL_PATH = "models/churn/best_churn_model.joblib"


# ==================================================
# DATAFRAME CACHE (dashboard / customer profile)
# ==================================================
# The dashboard and every customer-profile view used to call
# pd.read_csv(BUSINESS_DATA_PATH) (and the SHAP explanations file)
# from scratch on every single request, even though the underlying
# file only actually changes once per pipeline run -- typically
# minutes or longer between changes, versus every page view/poll.
#
# Cache keyed by the file's own mtime rather than a manual "clear the
# cache" call: a completed pipeline run always writes a new file with
# a new mtime (whether that's a fresh file the first time, or -- from
# the second run onward -- run_manager.py's atomic symlink swap
# pointing customers_business.csv at a different run's directory), so
# a stale entry is detected and replaced automatically the next time
# it's requested. This can't silently serve stale data after a run
# completes or after a fresh dataset is analyzed, and it doesn't need
# any invalidation call wired into the pipeline/run_manager code.
_CSV_CACHE_LOCK = threading.Lock()
_CSV_CACHE = {}  # path (str) -> (mtime, DataFrame)


def _load_cached_csv(path):
    """
    Read a CSV, reusing the in-memory copy from the last read as long
    as the file's mtime hasn't changed since. Raises the same
    FileNotFoundError/parser errors pd.read_csv would if the file is
    missing or unreadable, so callers don't need to change their error
    handling.
    """
    path = str(path)
    mtime = os.stat(path).st_mtime  # raises FileNotFoundError if missing, same as read_csv
    with _CSV_CACHE_LOCK:
        cached = _CSV_CACHE.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
    df = pd.read_csv(path)
    with _CSV_CACHE_LOCK:
        _CSV_CACHE[path] = (mtime, df)
    return df


# ==================================================
# ON-DEMAND, CACHED CUSTOMER-LEVEL SHAP (Task 3)
# ==================================================
# The pipeline used to pre-compute detailed SHAP explanations for up
# to 1,000 customers on every single run, whether or not anyone ever
# opened those profile pages (see shap_explainer.py's run() docstring).
# It now only computes global feature importance. A customer's own
# detailed explanation is instead computed the first time their
# profile page is opened, then cached here -- so opening the same
# customer's profile again reuses the cached result instead of
# recomputing it, and a new pipeline run (new model and/or new data)
# invalidates it automatically.

_SHAP_EXPLAINER_LOCK = threading.Lock()
_SHAP_EXPLAINER_CACHE = {}  # model path (str) -> (model_mtime, explainer, feature_names, model_pipeline)

_SHAP_EXPLANATION_LOCK = threading.Lock()
_SHAP_EXPLANATION_CACHE = {}  # customer_id (str) -> (model_mtime, data_mtime, explanation_records)


def _get_shap_explainer():
    """
    Build (once per churn-model version) and cache the SHAP explainer
    used for on-demand customer-level explanations. Rebuilt
    automatically the moment models/churn/best_churn_model.joblib
    changes -- i.e. the moment a pipeline run trains a new model or
    reuses a different one -- via the same mtime-keyed pattern as
    _load_cached_csv above.

    The background sample used to build the explainer comes from the
    cached business dataset: it only needs to be *representative* of
    the customer population, not the exact rows being explained (each
    customer's own detailed explanation always transforms that
    customer's own row, so its correctness only depends on using the
    model's own fitted preprocessor -- see shap_explainer.explain_customer_rows).
    """
    model_mtime = os.stat(CHURN_MODEL_PATH).st_mtime  # raises if the model doesn't exist yet
    with _SHAP_EXPLAINER_LOCK:
        cached = _SHAP_EXPLAINER_CACHE.get(CHURN_MODEL_PATH)
        if cached is not None and cached[0] == model_mtime:
            return cached[1], cached[2], cached[3]
    model_pipeline = joblib.load(CHURN_MODEL_PATH)
    background_df = _load_cached_csv(BUSINESS_DATA_PATH)
    background_features = shap_explainer.prepare_features(background_df, model_pipeline)
    explainer, _background_transformed, feature_names = shap_explainer.build_explainer(
        model_pipeline, background_features
    )
    with _SHAP_EXPLAINER_LOCK:
        _SHAP_EXPLAINER_CACHE[CHURN_MODEL_PATH] = (
            model_mtime,
            explainer,
            feature_names,
            model_pipeline,
        )
    return explainer, feature_names, model_pipeline


def _get_customer_shap_explanation(customer_id, customer_row_df):
    """
    On-demand, cached per-customer SHAP explanation. Computed the first
    time this customer's profile page is opened; reused on every later
    view of the same customer as long as neither the churn model nor
    the underlying business dataset has changed since (both checked by
    file mtime, so one new pipeline run invalidates this exactly when
    it invalidates the dashboard/profile CSV cache and the churn
    model-reuse check -- one consistent signal, not three separate
    ones to keep in sync by hand).

    Returns a list of {CustomerID, Feature, SHAP_Value, Impact} dicts,
    same shape build_customer_reasons already expects -- or an empty
    list if a SHAP explanation genuinely can't be computed (e.g. no
    model has been trained yet), so the profile page still renders
    without model-driven reasons rather than erroring.
    """
    try:
        model_mtime = os.stat(CHURN_MODEL_PATH).st_mtime
        data_mtime = os.stat(BUSINESS_DATA_PATH).st_mtime
    except OSError:
        return []
    cache_key = str(customer_id)
    with _SHAP_EXPLANATION_LOCK:
        cached = _SHAP_EXPLANATION_CACHE.get(cache_key)
        if cached is not None and cached[0] == model_mtime and cached[1] == data_mtime:
            return cached[2]
    try:
        explainer, feature_names, model_pipeline = _get_shap_explainer()
        explanation_df = shap_explainer.explain_customer_rows(
            customer_row_df, model_pipeline, explainer, feature_names
        )
    except Exception as error:
        print(f"Could not compute SHAP explanation for customer {customer_id}: {error}")
        return []
    records = explanation_df.to_dict("records")
    with _SHAP_EXPLANATION_LOCK:
        _SHAP_EXPLANATION_CACHE[cache_key] = (model_mtime, data_mtime, records)
    return records


# ==================================================
# UPLOAD SETTINGS
# ==================================================

UPLOAD_FOLDER = Path("data/raw/uploads")

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


ALLOWED_EXTENSIONS = {".csv", ".xlsx"}


# ==================================================
# REQUIRED CUSTOMER COLUMNS
# ==================================================

# CuStorA's own dataset schema. Note this is not a generic
# customer-data format -- an uploaded file needs these exact column
# names to run through the pipeline. A column-mapping step (letting a
# user match arbitrary column names to this schema) would make this
# genuinely generic, but that's future work, not part of this project.
REQUIRED_COLUMNS = [
    "CustomerID",
    "Age",
    "Gender",
    "RegistrationDate",
    "LastPurchaseDate",
    "TotalPurchases",
    "TotalSpent",
    "AverageOrderValue",
    "EngagementScore",
    "Complaints",
    "Returns",
    "DiscountUsage",
    "Churn",
]


# ==================================================
# HELPER FUNCTIONS
# ==================================================


REAL_DEMO_PATH = Path("data/external/online_retail_transactions_uci.csv")
LEDGER_PENDING_PATH = UPLOAD_FOLDER / "transactions_pending.csv"
LEDGER_STAGED_PATH = UPLOAD_FOLDER / "transactions_uploaded.csv"
WINDOW_CHOICES = [60, 90, 120, 180]
LEDGER_ROLES = ["customer_id", "date", "amount", "quantity", "unit_price", "invoice"]


def _stage(kind, path, **extra):
    """Remember what the next "Run analysis" click should run on."""
    STAGED_FILE.parent.mkdir(parents=True, exist_ok=True)
    STAGED_FILE.write_text(json.dumps({"kind": kind, "path": str(path), **extra}, default=str))


def _ledger_offer(df, filename, mapping=None, window_days=90):
    """The mapping-confirmation panel for a file that looks like a transaction ledger."""
    detected = transactions.detect_columns(df.columns, df)
    if not transactions.looks_like_transactions(df, detected):
        return None
    df.to_csv(LEDGER_PENDING_PATH, index=False)
    return {
        "filename": filename,
        "rows": len(df),
        "columns": list(df.columns),
        "mapping": mapping or detected,
        "roles": [(role, transactions.ROLE_LABELS[role]) for role in LEDGER_ROLES],
        "window_days": window_days,
        "window_choices": WINDOW_CHOICES,
    }


def allowed_file(filename):
    """
    Check whether uploaded file is CSV or XLSX.
    """
    extension = Path(filename).suffix.lower()
    return extension in ALLOWED_EXTENSIONS


def load_uploaded_file(file):
    """
    Read CSV or XLSX file into a pandas DataFrame.
    """
    extension = Path(file.filename).suffix.lower()
    if extension == ".csv":
        return pd.read_csv(file)
    if extension == ".xlsx":
        return pd.read_excel(file)
    raise ValueError("Only CSV and XLSX files are supported.")


def _build_mapping_offer(df, source_filename):
    """
    If some of REQUIRED_COLUMNS are missing but column_mapper.py finds
    plausible same-meaning matches under different names (e.g.
    "customer_id" -> CustomerID), stage the raw upload so the user can
    review and apply that mapping instead of having to rename columns
    themselves and re-upload. Returns None if column_mapper found
    nothing worth offering.
    """
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if not missing_required:
        return None
    mapping_result = column_mapper.suggest_mapping(df.columns)
    if not mapping_result["mapped"]:
        return None
    pending_path = UPLOAD_FOLDER / "customers_pending_mapping.csv"
    df.to_csv(pending_path, index=False)
    coverage = column_mapper.build_coverage_report(mapping_result, dataset_name=source_filename)
    return {
        "rows": coverage.to_dict("records"),
        "mapped_count": int((coverage["Status"] == "Mapped").sum()),
        "total_count": len(coverage),
        "fully_resolves": not mapping_result["unmapped_canonical"],
    }


def validate_uploaded_data(df):
    """
    Validate uploaded customer data.
    """
    errors = []
    warnings = []

    if df.empty:
        errors.append("The uploaded dataset is empty.")
        return errors, warnings

    # Required columns
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        errors.append("Missing required columns: " + ", ".join(missing_columns))

    # Duplicate customer IDs
    if "CustomerID" in df.columns:
        duplicate_ids = int(df["CustomerID"].duplicated().sum())
        if duplicate_ids > 0:
            warnings.append(f"{duplicate_ids} duplicate CustomerID value(s) found.")

    # Missing values
    missing_values = int(df.isna().sum().sum())
    if missing_values > 0:
        warnings.append(f"{missing_values} missing cell value(s) found.")

    # Numeric columns
    numeric_columns = [
        "Age",
        "TotalPurchases",
        "TotalSpent",
        "AverageOrderValue",
        "EngagementScore",
        "Complaints",
        "Returns",
        "DiscountUsage",
        "Churn",
    ]
    for column in numeric_columns:
        if column in df.columns:
            converted = pd.to_numeric(df[column], errors="coerce")
            invalid_count = int(converted.isna().sum())
            if invalid_count > 0:
                errors.append(f"{column} contains {invalid_count} non-numeric value(s).")

    # Churn validation
    if "Churn" in df.columns:
        churn_values = set(pd.to_numeric(df["Churn"], errors="coerce").dropna().unique())
        invalid_churn_values = [value for value in churn_values if value not in {0, 1}]
        if invalid_churn_values:
            errors.append("Churn must contain only 0 and 1.")
    return errors, warnings


def build_health_checks(df, errors, warnings):
    """
    Turn the flat errors/warnings lists (plus a couple of raw counts)
    into a structured dataset-health checklist: one row per check,
    each with a pass/warn/fail status and a short detail string.

    This exists purely for the upload page's "Dataset health" card --
    validate_uploaded_data() still drives the actual pass/fail logic
    for whether the pipeline is allowed to run; this just re-presents
    that same information so it reads like a checklist instead of a
    stack of banners.
    """
    checks = []
    row_count = len(df)
    column_count = len(df.columns)
    checks.append(
        {
            "label": f"{row_count:,} customers, {column_count} columns",
            "status": "pass" if row_count > 0 else "fail",
        }
    )
    missing_columns_error = next((e for e in errors if e.startswith("Missing required")), None)
    checks.append(
        {
            "label": (
                "Required columns present" if not missing_columns_error else missing_columns_error
            ),
            "status": "fail" if missing_columns_error else "pass",
        }
    )
    duplicate_warning = next((w for w in warnings if "duplicate" in w.lower()), None)
    checks.append(
        {
            "label": duplicate_warning or "No duplicate customer IDs",
            "status": "warn" if duplicate_warning else "pass",
        }
    )
    missing_values_warning = next((w for w in warnings if "missing" in w.lower()), None)
    checks.append(
        {
            "label": missing_values_warning or "No missing cell values",
            "status": "warn" if missing_values_warning else "pass",
        }
    )
    numeric_errors = [e for e in errors if "non-numeric" in e]
    if numeric_errors:
        for error in numeric_errors:
            checks.append({"label": error, "status": "fail"})
    else:
        checks.append({"label": "All numeric columns contain numbers", "status": "pass"})
    churn_error = next((e for e in errors if e.startswith("Churn must")), None)
    if "Churn" in df.columns:
        checks.append(
            {
                "label": churn_error or "Churn column is binary (0/1)",
                "status": "fail" if churn_error else "pass",
            }
        )
    return checks


def build_preview_table(df, n=10):
    """
    Build a preview of the uploaded dataset as plain columns/rows
    (not raw HTML), so the template can render it with normal Jinja
    auto-escaping instead of trusting a pre-rendered HTML string.
    """
    if df is None:
        return None
    preview_df = df.head(n)
    columns = [str(c) for c in preview_df.columns]
    rows = [
        [("" if pd.isna(v) else str(v)) for v in row] for row in preview_df.itertuples(index=False)
    ]
    return {"columns": columns, "rows": rows}


def render_upload_page(**overrides):
    """
    Render upload.html with sensible defaults, so each route only has
    to pass the handful of values that are different for its case.
    """
    context = {
        "required_columns": REQUIRED_COLUMNS,
        "errors": [],
        "warnings": [],
        "preview": None,
        "row_count": None,
        "column_count": None,
        "uploaded_filename": None,
        "upload_success": False,
        "saved_path": None,
        "health_checks": None,
        "mapping_offer": None,
        "ledger_offer": None,
        "ledger_summary": None,
        "real_demo_available": REAL_DEMO_PATH.exists(),
    }
    context.update(overrides)
    return render_template("upload.html", **context)


# ==================================================
# AUTHENTICATION -- fully removed (see the "AUTHENTICATION --
# DISABLED" note above). There's no /login or /logout route any more,
# and no "Log out" link left in the templates either -- there's
# nothing to log out of.
# ==================================================



# ==================================================
# BACKGROUND PIPELINE EXECUTION
# ==================================================
# /run-analysis used to run the full 12-stage pipeline synchronously
# inside the request, which could block the browser (and the request
# worker handling it) for a couple of minutes on a large dataset.
# Instead, the pipeline now runs on a background thread, and a
# polling page shows live per-stage progress from
# reports/pipeline_status.csv (written after every stage now, not just
# at the end -- see pipeline.py's run_pipeline()).
#
# PIPELINE_LOCK does double duty as the "only one run at a time" guard:
# this is a single-user demo, so a second POST while a run is already
# in flight doesn't start a second background thread -- it's just sent
# to the same progress page. (A proper task queue like RQ/Celery would
# be the way to support several people running analyses concurrently;
# that's more infrastructure than a single-user demo needs.)

PIPELINE_LOCK = threading.Lock()

PIPELINE_STATE_LOCK = threading.Lock()
PIPELINE_STATE = {
    "status": "idle",  # idle | running | completed | failed
    "run_id": None,
    "error": None,
}

STAGE_NAMES = list(PIPELINE_STAGES)

PIPELINE_STATUS_PATH = Path("reports/pipeline_status.csv")


def _set_pipeline_state(**updates):
    with PIPELINE_STATE_LOCK:
        PIPELINE_STATE.update(updates)


def _get_pipeline_state():
    with PIPELINE_STATE_LOCK:
        return dict(PIPELINE_STATE)


def _run_pipeline_in_background(uploaded_path, staged=None):
    """
    Runs on a background thread, started by /run-analysis. Holds
    PIPELINE_LOCK for the whole run so a second /run-analysis POST
    can't start a second run concurrently; always releases it,
    success or failure.
    """
    try:
        run_id = run_pipeline(uploaded_path, staged=staged)
        _set_pipeline_state(status="completed", run_id=run_id, error=None)
    except Exception as error:
        print("Background pipeline run failed:", error)
        _set_pipeline_state(status="failed", error=str(error))
    finally:
        PIPELINE_LOCK.release()


# ==================================================
# DASHBOARD
# ==================================================


def _gain_records(meta):
    """Held-out evidence for 'how far to trust the ranking' and the outreach planner (ledger runs only)."""
    try:
        gain_df = pd.read_csv("reports/churn/gain_curve.csv")
        if meta.get("data_kind") == "real_transactions":
            return gain_df[["ContactedPct", "ChurnersReachedPct", "HitRatePct"]].round(1).to_dict("records")
    except (OSError, ValueError, KeyError):
        pass
    return None


@app.route("/")
@login_required
def dashboard():
    df = _with_value_columns(_load_cached_csv(BUSINESS_DATA_PATH).copy())
    meta = _read_json(RUN_META_FILE) or {}
    window_days = int(meta.get("window_days") or 90)
    total = len(df)

    stake_total = float(df["RevenueAtStake"].sum())
    expected_loss = float(df["ExpectedLoss"].sum())
    expected_leavers = float(df["PredictedChurnProbability"].sum())
    # Where the expected loss sits by segment, and what to do about it.
    segments = []
    for name, part in df.groupby("SegmentName"):
        loss = float(part["ExpectedLoss"].sum())
        top_play = part["Recommendation"].mode()
        segments.append(
            {
                "name": name,
                "customers": int(len(part)),
                "avg_risk": float(part["PredictedChurnProbability"].mean() * 100),
                "expected_loss": loss,
                "loss_share": loss / expected_loss * 100 if expected_loss else 0.0,
                "play": top_play.iloc[0] if len(top_play) else "",
            }
        )
    segments.sort(key=lambda s: s["expected_loss"], reverse=True)

    gain = _gain_records(meta)

    try:
        dataset_updated = datetime.fromtimestamp(Path(BUSINESS_DATA_PATH).stat().st_mtime).strftime("%b %d, %Y")
    except OSError:
        dataset_updated = "unknown"

    reached = set(_reached_records())
    todo = df[~df["CustomerID"].astype(str).isin(reached)]
    priority = todo["BusinessPriority"]
    return render_template(
        "dashboard.html",
        total_customers=total,
        window_days=window_days,
        stake_total=stake_total,
        expected_loss=expected_loss,
        loss_pct=(expected_loss / stake_total * 100) if stake_total else 0.0,
        expected_leavers=expected_leavers,
        leavers_pct=(expected_leavers / total * 100) if total else 0.0,
        act_now=int(priority.isin(["Critical Business Priority", "High Business Priority"]).sum()),
        critical_business=int((priority == "Critical Business Priority").sum()),
        quad_meta={key: {"title": info["title"], "play": info["play"]} for key, info in QUADRANTS.items()},
        reached_total=len(reached),
        segments=segments,
        drivers=_top_drivers(5),
        gain=gain,
        test_auc=meta.get("test_roc_auc"),
        states=todo["CustomerState"].value_counts().to_dict(),  # customers still to contact, so the counts match the queue
        dataset_updated=dataset_updated,
        avg_health_score=float(df["RelationshipHealthScore"].mean()) if total else 0.0,
    )


# ==================================================
# VALUE COLUMNS + QUADRANTS (shared by the dashboard, the matrix and the list)
# ==================================================

QUADRANTS = {
    "save_now": {
        "title": "Save now",
        "play": "High value and likely to leave: personal outreach this week.",
        "tone": "critical",
    },
    "protect": {
        "title": "Protect",
        "play": "High value, currently loyal: keep the relationship warm.",
        "tone": "healthy",
    },
    "automate": {
        "title": "Win back cheaply",
        "play": "Likely to leave but modest value: automated offers, no calls.",
        "tone": "watch",
    },
    "healthy": {
        "title": "Leave alone",
        "play": "Modest value and loyal: standard service.",
        "tone": "muted",
    },
}


def _with_value_columns(df):
    """Older analysis files predate RevenueAtStake / ExpectedLoss; derive them if absent."""
    if "RevenueAtStake" not in df.columns:
        from src.business import clv_metrics

        window = int((_read_json(RUN_META_FILE) or {}).get("window_days") or 90)
        df["RevenueAtStake"] = df.apply(lambda r: clv_metrics.revenue_at_stake(r, window), axis=1).round(2)
    if "ExpectedLoss" not in df.columns:
        df["ExpectedLoss"] = (df["RevenueAtStake"] * df["PredictedChurnProbability"]).round(2)
    return df


def _quadrant_cuts(df):
    """(risk cut, value cut): the Medium-risk line and the top-third-by-stake line."""
    cuts = _read_json("reports/churn/decision_thresholds.json") or {}
    risk_cut = cuts.get("medium_risk_threshold")
    if risk_cut is None:
        risk_cut = float(df["PredictedChurnProbability"].quantile(0.65))
    value_cut = float(df["RevenueAtStake"].quantile(0.67))
    return float(risk_cut), value_cut


def _quadrants(df):
    risk_cut, value_cut = _quadrant_cuts(df)
    risky = df["PredictedChurnProbability"] >= risk_cut
    valuable = df["RevenueAtStake"] >= value_cut
    out = pd.Series("healthy", index=df.index)
    out[risky & valuable] = "save_now"
    out[~risky & valuable] = "protect"
    out[risky & ~valuable] = "automate"
    return out


def _quadrant_summary(df, reached):
    """The four groups for the Risk x value page, counting only customers not yet reached.
    The cut lines come from EVERYONE, so reaching a customer never moves anybody else between groups."""
    quad = _quadrants(df)
    keep = ~df["CustomerID"].astype(str).isin(reached)
    active, quad = df[keep], quad[keep]
    total_loss = float(active["ExpectedLoss"].sum())
    groups = []
    for key, info in QUADRANTS.items():
        part = active[quad == key]
        loss = float(part["ExpectedLoss"].sum())
        groups.append(
            {
                "key": key,
                "title": info["title"],
                "play": info["play"],
                "tone": info["tone"],
                "customers": int(len(part)),
                "expected_loss": loss,
                "loss_share": loss / total_loss * 100 if total_loss else 0.0,
            }
        )
    return groups


# ==================================================
# CUSTOMER LIST (state / quadrant filter, search, sort, paging over ALL customers)
# ==================================================

PRIORITY_RANK = {
    "Critical Business Priority": 0,
    "High Business Priority": 1,
    "Medium Business Priority": 2,
    "Normal Business Priority": 3,
}
LIST_FIELDS = [
    "CustomerID", "CustomerState", "PredictedChurnProbability", "ChurnRisk",
    "RelationshipHealth", "EstimatedCLV", "RevenueAtStake", "ExpectedLoss",
    "RecommendationPriority", "Recommendation", "SegmentName", "BusinessPriority",
]
SORTS = {
    "priority": None,
    "loss": ("ExpectedLoss", False),
    "risk": ("PredictedChurnProbability", False),
    "stake": ("RevenueAtStake", False),
}


def _filtered_ranked(args):
    """Apply the list filters + sort that the table, the planner and the export share."""
    full = _with_value_columns(_load_cached_csv(BUSINESS_DATA_PATH))
    # Customers already reached are not part of the to-do list. The group cut lines still
    # come from everyone, so reaching someone never moves another customer between groups.
    keep = ~full["CustomerID"].astype(str).isin(set(_reached_records()))
    df = full[keep]
    state = args.get("state", "__all__")
    quadrant = args.get("quadrant", "__all__")
    term = args.get("q", "").strip().lower()
    sort = args.get("sort", "priority")
    if sort not in SORTS:
        sort = "priority"
    if quadrant in QUADRANTS:
        df = df[_quadrants(full)[keep] == quadrant]
    if state != "__all__":
        df = df[df["CustomerState"] == state]
    if term:
        df = df[df["CustomerID"].astype(str).str.lower().str.contains(term, regex=False)]
    if SORTS[sort] is None:
        ranked = df.assign(_rank=df["BusinessPriority"].map(PRIORITY_RANK).fillna(99)).sort_values(
            by=["_rank", "ExpectedLoss"], ascending=[True, False]
        )
    else:
        column, ascending = SORTS[sort]
        ranked = df.sort_values(by=column, ascending=ascending)
    return ranked


def _int_arg(name, default, low, high):
    try:
        return max(low, min(int(request.args.get(name, default)), high))
    except ValueError:
        return default


@app.route("/customers/list.json")
@login_required
def customers_list_json():
    """Server-side filtering over the WHOLE customer base (not just the first rows)."""
    ranked = _filtered_ranked(request.args)
    limit = _int_arg("limit", 50, 1, 200)
    offset = _int_arg("offset", 0, 0, 10**7)
    rows = [
        {key: _json_safe(value) for key, value in record.items()}
        for record in ranked.iloc[offset : offset + limit][LIST_FIELDS].to_dict("records")
    ]
    return jsonify({"total": int(len(ranked)), "shown": len(rows), "offset": offset, "rows": rows})


def _csv_safe(value):
    """Stop spreadsheet apps from executing a cell that starts with = + - @ (uploaded data is untrusted)."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


@app.route("/customers/export.csv")
@login_required
def customers_export_csv():
    """A call list: the top N customers under the current filters, ready for a spreadsheet."""
    import csv
    import io

    ranked = _filtered_ranked(request.args)
    limit = _int_arg("limit", 100, 1, 5000)
    top = ranked.head(limit)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "CustomerID", "ChurnProbabilityPct", "RevenueAtStake", "ExpectedLoss", "CustomerState",
        "Segment", "RecommendedAction", "WhyThisAction", "DaysSinceLastPurchase",
    ])
    for _, r in top.iterrows():
        writer.writerow([
            _csv_safe(r["CustomerID"]),
            round(float(r["PredictedChurnProbability"]) * 100, 1),
            round(float(r["RevenueAtStake"]), 2),
            round(float(r["ExpectedLoss"]), 2),
            _csv_safe(r["CustomerState"]),
            _csv_safe(r["SegmentName"]),
            _csv_safe(r.get("Recommendation", "")),
            _csv_safe(r.get("RecommendationEvidence", "")),
            "" if pd.isna(r.get("Recency")) else int(r["Recency"]),
        ])
    response = app.response_class(buffer.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=custora_call_list.csv"
    return response


_STATE_FILLS = {
    "critical": "F8D7DA",  # Critical Retention / Critical Health
    "high": "FDE2C8",      # High Risk
    "watch": "FFF3C4",     # Watch
    "stable": "D9EFD9",    # Stable
}


def _state_fill(state):
    text = str(state).lower()
    for key, colour in _STATE_FILLS.items():
        if text.startswith(key):
            return colour
    return None


@app.route("/customers/export.xlsx")
@login_required
def customers_export_xlsx():
    """The same call list as the CSV, but as a formatted Excel workbook people can use directly."""
    import io

    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    ranked = _filtered_ranked(request.args)
    limit = _int_arg("limit", 100, 1, 5000)
    top = ranked.head(limit)
    meta = _read_json(RUN_META_FILE) or {}
    currency = meta.get("currency") or "\u20b9"
    window = int(meta.get("window_days") or 90)

    font = "Arial"
    thin = Side(style="thin", color="D0D5DD")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    wb = Workbook()
    ws = wb.active
    ws.title = "Call list"
    headers = [
        ("#", 6), ("Customer ID", 14), ("Churn risk", 11), (f"Revenue at stake ({currency})", 18),
        (f"Expected loss ({currency})", 18), ("State", 20), ("Segment", 32),
        ("Recommended action", 44), ("Why this action", 60), ("Days since last purchase", 16),
        ("Called? / notes", 28),
    ]
    for col, (name, width) in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col, value=name)
        cell.font = Font(name=font, bold=True, color="FFFFFF", size=11)
        cell.fill = PatternFill("solid", fgColor="1F4E46")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.row_dimensions[1].height = 34

    def put_text(cell, value):
        # Uploaded data is untrusted: force text so a value like =HYPERLINK(...) is never a formula.
        cell.value = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
        cell.data_type = "s"

    for position, (_, r) in enumerate(top.iterrows(), start=1):
        row = position + 1
        ws.cell(row=row, column=1, value=position)
        cid = str(r["CustomerID"])
        id_cell = ws.cell(row=row, column=2)
        if cid.isdigit():
            id_cell.value = int(cid)
            id_cell.number_format = "0"
        else:
            put_text(id_cell, cid)
        ws.cell(row=row, column=3, value=float(r["PredictedChurnProbability"])).number_format = "0.0%"
        ws.cell(row=row, column=4, value=round(float(r["RevenueAtStake"]), 2)).number_format = f'"{currency}"#,##0'
        ws.cell(row=row, column=5, value=round(float(r["ExpectedLoss"]), 2)).number_format = f'"{currency}"#,##0'
        put_text(ws.cell(row=row, column=6), r["CustomerState"])
        put_text(ws.cell(row=row, column=7), r["SegmentName"])
        put_text(ws.cell(row=row, column=8), r.get("Recommendation", ""))
        put_text(ws.cell(row=row, column=9), r.get("RecommendationEvidence", ""))
        recency = r.get("Recency")
        ws.cell(row=row, column=10, value=None if pd.isna(recency) else int(recency)).number_format = "#,##0"
        ws.cell(row=row, column=11, value=None)

        fill = _state_fill(r["CustomerState"])
        for col in range(1, len(headers) + 1):
            cell = ws.cell(row=row, column=col)
            cell.font = Font(name=font, size=10)
            cell.border = border
            wrap = col in (8, 9, 11)
            cell.alignment = Alignment(
                vertical="top",
                wrap_text=wrap,
                horizontal="left" if col in (2, 6, 7, 8, 9, 11) else "center" if col in (1, 3, 10) else "right",
            )
        if fill:
            ws.cell(row=row, column=6).fill = PatternFill("solid", fgColor=fill)

    last_row = max(len(top) + 1, 2)
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{last_row}"
    ws.sheet_view.zoomScale = 100
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_title_rows = "1:1"

    about = wb.create_sheet("About this list")
    about.column_dimensions["A"].width = 26
    about.column_dimensions["B"].width = 80
    filters = {
        "Sorted by": request.args.get("sort", "priority"),
        "State filter": request.args.get("state", "__all__").replace("__all__", "All"),
        "Group filter": request.args.get("quadrant", "__all__").replace("__all__", "All"),
    }
    lines = [
        ("CuStorA call list", ""),
        ("Customers in this list", len(top)),
        ("Data source", meta.get("source_label") or "Not recorded"),
        ("Churn window", f"Next {window} days"),
        ("Churn risk", "Modelled chance the customer stops buying within the window."),
        ("Revenue at stake", f"Normal spend per {window}-day window, in {currency}."),
        ("Expected loss", "Churn risk x revenue at stake. The list is ranked by this unless stated below."),
        *filters.items(),
        ("Note", "Scores are model estimates to prioritise outreach, not certainties."),
        ("Why Stable customers appear", "Ranking is by expected loss (risk x revenue), so a high-value customer can rank high even when their state is Stable."),
    ]
    if str(meta.get("data_kind", "")).startswith("synthetic"):
        lines.append(("Warning", "Synthetic demo data. These customers are not real."))
    for i, (label, value) in enumerate(lines, start=1):
        a = about.cell(row=i, column=1, value=label)
        b = about.cell(row=i, column=2)
        put_text(b, value) if isinstance(value, str) else setattr(b, "value", value)
        a.font = Font(name=font, bold=True, size=14 if i == 1 else 10)
        b.font = Font(name=font, size=10)
        b.alignment = Alignment(wrap_text=True, vertical="top", horizontal="left")
        a.alignment = Alignment(vertical="top")

    buffer = io.BytesIO()
    wb.save(buffer)
    response = app.response_class(
        buffer.getvalue(),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    response.headers["Content-Disposition"] = "attachment; filename=custora_call_list.xlsx"
    return response


# ==================================================
# RISK x VALUE MATRIX (data for the dashboard chart)
# ==================================================

MATRIX_MAX_POINTS = 6000


@app.route("/api/matrix.json")
@login_required
def matrix_json():
    """
    Every customer as (churn probability, revenue at stake). Compact parallel
    arrays rather than one object per customer. Above MATRIX_MAX_POINTS the
    lowest-stakes customers are thinned so the chart stays fast; the biggest
    expected losses are always kept.
    """
    full = _with_value_columns(_load_cached_csv(BUSINESS_DATA_PATH))
    risk_cut, value_cut = _quadrant_cuts(full)
    keep = ~full["CustomerID"].astype(str).isin(set(_reached_records()))
    quad = _quadrants(full)[keep]
    df = full[keep]  # customers already reached are left off the chart and out of the planner
    keys = list(QUADRANTS)
    frame = pd.DataFrame(
        {
            "id": df["CustomerID"].astype(str),
            "p": df["PredictedChurnProbability"].round(3),
            "v": df["RevenueAtStake"].round(0),
            "l": df["ExpectedLoss"].round(0),
            "q": quad.map({k: i for i, k in enumerate(keys)}),
        }
    )
    truncated = len(frame) > MATRIX_MAX_POINTS
    if truncated:
        keep = frame.nlargest(MATRIX_MAX_POINTS // 2, "l").index
        rest = frame.drop(keep)
        frame = pd.concat([frame.loc[keep], rest.sample(MATRIX_MAX_POINTS - len(keep), random_state=1)])
    return jsonify(
        {
            "n": int(len(df)),
            "plotted": int(len(frame)),
            "risk_cut": round(risk_cut, 4),
            "value_cut": round(value_cut, 2),
            "quadrants": keys,
            "id": frame["id"].tolist(),
            "p": frame["p"].tolist(),
            "v": frame["v"].tolist(),
            "l": frame["l"].tolist(),
            "q": frame["q"].astype(int).tolist(),
        }
    )


# ==================================================
# ANALYSE PAGES: Risk x value, Outreach planner, Customer clusters
# ==================================================


@app.route("/risk-value")
@login_required
def risk_value_page():
    df = _with_value_columns(_load_cached_csv(BUSINESS_DATA_PATH))
    return render_template(
        "risk_value.html",
        quadrants=_quadrant_summary(df, set(_reached_records())),
        reached_total=len(_reached_records()),
        quad_meta={key: {"title": info["title"], "play": info["play"]} for key, info in QUADRANTS.items()},
    )


@app.route("/outreach-planner")
@login_required
def outreach_planner_page():
    meta = _read_json(RUN_META_FILE) or {}
    return render_template(
        "planner.html",
        gain=_gain_records(meta),
        window_days=int(meta.get("window_days") or 90),
        reached_total=len(_reached_records()),
    )


CLUSTER_MAX_POINTS = 6000


@app.route("/clusters")
@login_required
def clusters_page():
    return render_template("clusters.html")


@app.route("/api/clusters.json")
@login_required
def clusters_json():
    """Every customer as a point on the PCA map (PC1, PC2), coloured by segment, plus a profile per segment."""
    df = _load_cached_csv(BUSINESS_DATA_PATH)
    try:
        pca_df = pd.read_csv("data/processed/customers_pca.csv", usecols=["CustomerID", "PC1", "PC2"])
    except (OSError, ValueError):
        return jsonify({"available": False})
    pca_df["CustomerID"] = pca_df["CustomerID"].astype(str)

    frame = df[["CustomerID", "SegmentName", "PredictedChurnProbability"]].copy()
    frame["CustomerID"] = frame["CustomerID"].astype(str)
    frame = frame.merge(pca_df, on="CustomerID", how="inner").dropna(subset=["PC1", "PC2", "SegmentName"])

    seg_df = df.dropna(subset=["SegmentName"])
    segments = []
    for name, part in seg_df.groupby("SegmentName"):
        segments.append(
            {
                "name": str(name),
                "customers": int(len(part)),
                "recency": _json_safe(round(float(pd.to_numeric(part["Recency"], errors="coerce").median()), 0)),
                "frequency": _json_safe(round(float(pd.to_numeric(part["Frequency"], errors="coerce").median()), 1)),
                "monetary": _json_safe(round(float(pd.to_numeric(part["Monetary"], errors="coerce").median()), 0)),
                "avg_risk": round(float(part["PredictedChurnProbability"].mean()) * 100, 1),
            }
        )
    segments.sort(key=lambda seg: seg["customers"], reverse=True)
    index = {seg["name"]: i for i, seg in enumerate(segments)}

    if len(frame) > CLUSTER_MAX_POINTS:
        frame = frame.sample(CLUSTER_MAX_POINTS, random_state=1)
    return jsonify(
        {
            "available": True,
            "n": int(len(seg_df)),
            "plotted": int(len(frame)),
            "segments": segments,
            "id": frame["CustomerID"].tolist(),
            "x": frame["PC1"].round(3).tolist(),
            "y": frame["PC2"].round(3).tolist(),
            "c": frame["SegmentName"].map(index).astype(int).tolist(),
            "p": frame["PredictedChurnProbability"].round(3).tolist(),
        }
    )


# ==================================================
# REACHED CUSTOMERS (done with outreach: out of the queue, listed here)
# ==================================================


@app.route("/reached")
@login_required
def reached_page():
    df = _with_value_columns(_load_cached_csv(BUSINESS_DATA_PATH))
    known = set(df["CustomerID"].astype(str))
    records = _reached_records()
    rows = []
    for customer_id, record in sorted(records.items(), key=lambda kv: kv[1].get("reached_at", ""), reverse=True):
        snap = record.get("snapshot") or {}
        rows.append(
            {
                "id": customer_id,
                "reached_at": record.get("reached_at", ""),
                "risk": snap.get("PredictedChurnProbability"),
                "stake": snap.get("RevenueAtStake"),
                "loss": snap.get("ExpectedLoss"),
                "state": snap.get("CustomerState"),
                "segment": snap.get("SegmentName"),
                "action": snap.get("Recommendation"),
                "in_data": customer_id in known,
            }
        )
    covered = sum(float(r["loss"] or 0) for r in rows)
    return render_template("reached.html", rows=rows, covered=covered, total_customers=len(df))


@app.route("/customers/<customer_id>/reached", methods=["POST"])
@login_required
def mark_customer_reached(customer_id):
    """Your team has reached this customer: take them off the to-do list and file them under Reached."""
    df = _with_value_columns(_load_cached_csv(BUSINESS_DATA_PATH))
    rows = df[df["CustomerID"].astype(str) == str(customer_id)]
    if rows.empty:
        abort(404)
    record = rows.iloc[0].to_dict()
    snapshot = {k: _json_safe(v) for k, v in record.items() if k in reached_store.SNAPSHOT_FIELDS}
    stored = reached_store.mark(customer_id, snapshot, _dataset_fingerprint())
    return jsonify({"ok": True, "reached_at": stored["reached_at"], "reached_count": len(_reached_records())})


@app.route("/customers/<customer_id>/unreached", methods=["POST"])
@login_required
def unmark_customer_reached(customer_id):
    """Put a customer back on the to-do list (for example, marked by mistake)."""
    existed = reached_store.unmark(customer_id, _dataset_fingerprint())
    return jsonify({"ok": True, "was_reached": existed, "reached_count": len(_reached_records())})


# ==================================================
# MODEL EVIDENCE ("how do we know it works?")
# ==================================================

# Independent real-data checks that ship with the project. Numbers are read
# from the report files each script wrote, never typed in by hand here.
EXTERNAL_CHECKS = [
    {
        "name": "UCI Online Retail",
        "what": "397,884 real purchase lines by 4,338 identified customers of a UK gift retailer (Dec 2010 - Dec 2011)",
        "split": "Time-based: trained on earlier months, tested on a later month",
        "source": "archive.ics.uci.edu/dataset/352/online+retail (Chen, Sain & Guo, 2012)",
        "kind": "json",
        "path": "reports/external_validation_temporal_online_retail/model_comparison.json",
    },
    {
        "name": "IBM Telco Customer Churn",
        "what": "7,043 real telecom customers with real churn labels",
        "split": "Random 20% held-out test set",
        "source": "github.com/IBM/telco-customer-churn-on-icp4d",
        "kind": "csv",
        "path": "reports/external_validation/churn_reports/final_test_evaluation.csv",
    },
    {
        "name": "Kaggle E-Commerce Churn",
        "what": "5,630 real e-commerce customers with real churn labels",
        "split": "Random 20% held-out test set",
        "source": "Kaggle E-Commerce Customer Churn (via github.com/anvarnarz/praktikum_datasets)",
        "kind": "csv",
        "path": "reports/external_validation_ecommerce/churn_reports/final_test_evaluation.csv",
    },
]


def _external_results():
    rows = []
    for check in EXTERNAL_CHECKS:
        row = {k: check[k] for k in ("name", "what", "split", "source")}
        row.update(roc_auc=None, detail="", test_rows=None)
        try:
            if check["kind"] == "json":
                results = json.loads(Path(check["path"]).read_text())
                name, best = max(results.items(), key=lambda kv: kv[1]["test"]["roc_auc"])
                row.update(
                    roc_auc=best["test"]["roc_auc"],
                    detail=f"{name}: PR-AUC {best['test']['pr_auc']}, top-10% lift {best['test']['lift_at_10pct']:.1f}x",
                )
            else:
                frame = pd.read_csv(check["path"]).iloc[0]
                row.update(
                    roc_auc=round(float(frame["ROC-AUC"]), 3),
                    test_rows=int(frame["TestRows"]),
                    detail=str(frame["Model"]),
                )
        except (OSError, ValueError, KeyError, IndexError):
            pass
        rows.append(row)
    return rows


def _top_drivers(n=8):
    try:
        importance = pd.read_csv("reports/shap/global_feature_importance.csv")
    except OSError:
        return []
    drivers = {}
    for _, row in importance.iterrows():
        base = base_feature_name(row["Feature"])
        info = reason_translator.FEATURE_CONCEPTS.get(base, {})
        label = info.get("label", base)
        drivers[label] = drivers.get(label, 0) + float(row["MeanAbsoluteSHAP"])
    ranked = sorted(drivers.items(), key=lambda kv: kv[1], reverse=True)[:n]
    top = ranked[0][1] if ranked else 1
    return [{"label": k, "share": round(v / top * 100)} for k, v in ranked if v > 0]


@app.route("/evidence")
@login_required
def evidence():
    meta = _read_json(RUN_META_FILE) or {}
    backtest = _read_json("reports/churn/backtest_summary.json") if meta.get("data_kind") == "real_transactions" else None
    deciles = None
    if backtest:
        try:
            deciles = pd.read_csv("reports/churn/test_risk_deciles.csv").to_dict("records")
        except OSError:
            deciles = None
    gain = None
    if backtest:
        try:
            gain = pd.read_csv("reports/churn/gain_curve.csv").round(1).to_dict("records")
        except OSError:
            gain = None
    coverage = None
    if COVERAGE_FILE.exists():
        coverage = pd.read_csv(COVERAGE_FILE).to_dict("records")
    synthetic_row = None
    if meta.get("data_kind") in ("synthetic", "customer_table"):
        try:
            synthetic_row = pd.read_csv("reports/churn/final_test_evaluation.csv").iloc[0].to_dict()
        except (OSError, IndexError):
            synthetic_row = None
    return render_template(
        "evidence.html",
        backtest=backtest,
        deciles=deciles,
        gain=gain,
        coverage=coverage,
        synthetic_row=synthetic_row,
        external=_external_results(),
        drivers=_top_drivers(),
    )


@app.route("/reports/<path:filename>")
def report_image(filename):
    """Serve report charts (PNG only) so the evidence page can show them."""
    if not filename.lower().endswith(".png"):
        abort(404)
    return send_from_directory(Path(app.root_path) / "reports", filename)


# ==================================================
# CUSTOMER PROFILE
# ==================================================


def _reason_reference(df):
    """Typical (median) value of each explainable feature, so reasons can be sanity-checked.
    Also makes sure reason text uses this run's currency (JSON endpoints skip the template hook)."""
    reason_translator.CURRENCY_SYMBOL = (_read_json(RUN_META_FILE) or {}).get("currency") or "\u20b9"
    names = [c for c in reason_translator.FEATURE_CONCEPTS if c in df.columns]
    return {c: float(pd.to_numeric(df[c], errors="coerce").median()) for c in names}


@app.route("/customer/<customer_id>")
@login_required
def customer_profile(customer_id):
    df = _load_cached_csv(BUSINESS_DATA_PATH)
    customer_rows = df[df["CustomerID"].astype(str) == str(customer_id)]
    if customer_rows.empty:
        abort(404)
    customer = customer_rows.iloc[0].to_dict()

    # On-demand, cached SHAP explanation for this one customer (see
    # _get_customer_shap_explanation above) -- computed the first time
    # this profile is opened, reused on every later view until a new
    # pipeline run changes the model or the data. Returns [] (rather
    # than raising) if no model exists yet or SHAP can't be computed.
    shap_explanations = _get_customer_shap_explanation(customer_id, customer_rows.iloc[[0]])

    churn_reasons = build_customer_reasons(
        customer, shap_explanations, max_reasons=3, reference=_reason_reference(df)
    )
    return render_template(
        "customer/profile.html",
        customer=customer,
        churn_reasons=churn_reasons,
        reached=_reached_records().get(str(customer_id)),
    )


def _json_safe(value):
    """
    Coerce a single customer-row value to something json.dumps can
    handle. Values coming out of DataFrame.to_dict() are frequently
    numpy scalar types (numpy.int64, numpy.float64, numpy.bool_)
    rather than native Python types, and NaN needs to become null
    rather than being sent as the non-standard token `NaN`. This
    mirrors, in JSON form, the same customer dict the existing
    customer_profile page already renders -- no new calculations,
    just a safe-to-serialize copy of the same values.
    """
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


@app.route("/customer/<customer_id>/panel.json")
@login_required
def customer_profile_panel(customer_id):
    """
    JSON companion to /customer/<id>, used by the dashboard's quick-
    view drawer so a row can be inspected without leaving the priority
    queue. Reuses the exact same lookup, SHAP explanation, and reason-
    translation logic as the full profile page below -- this is a
    read-only view of that same data, not a new calculation, and the
    full-page route above is unchanged and still the source of truth
    for the "View full customer profile" link.
    """
    df = _load_cached_csv(BUSINESS_DATA_PATH)
    customer_rows = df[df["CustomerID"].astype(str) == str(customer_id)]
    if customer_rows.empty:
        abort(404)
    customer = customer_rows.iloc[0].to_dict()
    shap_explanations = _get_customer_shap_explanation(customer_id, customer_rows.iloc[[0]])
    churn_reasons = build_customer_reasons(
        customer, shap_explanations, max_reasons=3, reference=_reason_reference(df)
    )
    reasons_with_magnitude = churn_reasons  # each reason already carries its |SHAP| magnitude

    customer_json = {k: _json_safe(v) for k, v in customer.items()}
    return jsonify(
        {
            "customer": customer_json,
            "churn_reasons": reasons_with_magnitude,
            "profile_url": url_for("customer_profile", customer_id=customer_id),
        }
    )


# ==================================================
# UPLOAD DATASET
# ==================================================


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload_dataset():
    if request.method == "GET":
        return render_upload_page()

    uploaded_file = request.files.get("dataset")
    if uploaded_file is None or uploaded_file.filename == "":
        flash("Please select a dataset file.", "error")
        return redirect(url_for("upload_dataset"))

    if not allowed_file(uploaded_file.filename):
        flash("Invalid file type. Please upload CSV or XLSX.", "error")
        return redirect(url_for("upload_dataset"))

    try:
        df = load_uploaded_file(uploaded_file)
    except Exception as error:
        return render_upload_page(
            errors=[f"Could not read the file: {error}"],
            uploaded_filename=uploaded_file.filename,
        )

    # A transaction ledger (one row per order line) is the most common thing a
    # company actually has -- offer to map it instead of rejecting it.
    if not all(c in df.columns for c in REQUIRED_COLUMNS):
        offer = _ledger_offer(df, uploaded_file.filename)
        if offer:
            return render_upload_page(
                ledger_offer=offer,
                preview=build_preview_table(df),
                row_count=len(df),
                column_count=len(df.columns),
                uploaded_filename=uploaded_file.filename,
            )

    errors, warnings = validate_uploaded_data(df)

    # Validation failed: show the errors, offering a column mapping if one exists
    if errors:
        mapping_offer = _build_mapping_offer(df, uploaded_file.filename)
        return render_upload_page(
            errors=errors,
            warnings=warnings,
            preview=build_preview_table(df),
            row_count=len(df),
            column_count=len(df.columns),
            uploaded_filename=uploaded_file.filename,
            health_checks=build_health_checks(df, errors, warnings),
            mapping_offer=mapping_offer,
        )

    saved_path = UPLOAD_FOLDER / "customers_uploaded.csv"
    df.to_csv(saved_path, index=False)
    _stage("customers", saved_path, data_kind="customer_table", label="Uploaded customer table")
    return render_upload_page(
        warnings=warnings,
        preview=build_preview_table(df),
        row_count=len(df),
        column_count=len(df.columns),
        uploaded_filename=uploaded_file.filename,
        upload_success=True,
        saved_path=str(saved_path),
        health_checks=build_health_checks(df, errors, warnings),
    )


# ==================================================
# APPLY A SUGGESTED COLUMN MAPPING
# ==================================================


@app.route("/upload/apply-mapping", methods=["POST"])
@login_required
def apply_column_mapping():
    """
    Confirms and applies the column mapping offered on the upload page
    (see _build_mapping_offer / src/data/column_mapper.py) against the
    raw file that was staged when the mismatch was first detected.
    """
    pending_path = UPLOAD_FOLDER / "customers_pending_mapping.csv"
    if not pending_path.exists():
        flash("No pending column mapping found. Please upload the file again.", "error")
        return redirect(url_for("upload_dataset"))
    raw_df = pd.read_csv(pending_path)
    mapping_result = column_mapper.suggest_mapping(raw_df.columns)
    mapped_df = column_mapper.apply_mapping(raw_df, mapping_result)
    errors, warnings = validate_uploaded_data(mapped_df)
    display_name = "Mapped dataset"
    if errors:

        # Some canonical fields still have no equivalent source column
        # -- that's a real "this data doesn't exist" gap (see
        # column_mapper.py's docstring), not something to guess at, so
        # it's reported the same way an original upload's missing
        # columns would be.
        return render_upload_page(
            errors=errors,
            warnings=warnings,
            preview=build_preview_table(mapped_df),
            row_count=len(mapped_df),
            column_count=len(mapped_df.columns),
            uploaded_filename=display_name,
            health_checks=build_health_checks(mapped_df, errors, warnings),
        )
    saved_path = UPLOAD_FOLDER / "customers_uploaded.csv"
    mapped_df.to_csv(saved_path, index=False)
    _stage("customers", saved_path, data_kind="customer_table", label="Uploaded customer table")
    pending_path.unlink(missing_ok=True)
    flash("Column mapping applied. Your dataset is ready to run.", "success")
    return render_upload_page(
        warnings=warnings,
        preview=build_preview_table(mapped_df),
        row_count=len(mapped_df),
        column_count=len(mapped_df.columns),
        uploaded_filename=display_name,
        upload_success=True,
        saved_path=str(saved_path),
        health_checks=build_health_checks(mapped_df, errors, warnings),
    )


# ==================================================
# TRANSACTION LEDGER: CONFIRM COLUMN MAPPING
# ==================================================


@app.route("/upload/confirm-ledger", methods=["POST"])
@login_required
def confirm_ledger():
    """Validate the user's choice of id/date/amount columns and stage the ledger."""
    if not LEDGER_PENDING_PATH.exists():
        flash("No pending file found. Please upload it again.", "error")
        return redirect(url_for("upload_dataset"))
    df = pd.read_csv(LEDGER_PENDING_PATH, low_memory=False)
    mapping = {role: (request.form.get(role) or None) for role in LEDGER_ROLES}
    try:
        window_days = int(request.form.get("window_days", 90))
    except ValueError:
        window_days = 90
    if window_days not in WINDOW_CHOICES:
        window_days = 90
    currency = (request.form.get("currency") or "\u20b9").strip()[:3]
    filename = request.form.get("filename") or "Transaction file"

    def back_with(errors):
        return render_upload_page(
            errors=errors,
            ledger_offer=_ledger_offer(df, filename, mapping, window_days),
            preview=build_preview_table(df),
            row_count=len(df),
            column_count=len(df.columns),
            uploaded_filename=filename,
        )

    try:
        tx, report = transactions.prepare_transactions(df, mapping)
    except ValueError as error:
        return back_with([str(error)])
    needed = window_days * 2 + 90
    if report["span_days"] < needed or len(transactions.plan_cutoffs(tx, window_days)) < 3:
        return back_with(
            [
                f"This file covers {report['span_days']} days. To train and fairly test a model that "
                f"predicts churn over {window_days} days it needs roughly {needed}+ days of history. "
                "Choose a shorter churn window or use a longer export."
            ]
        )
    df.to_csv(LEDGER_STAGED_PATH, index=False)
    LEDGER_PENDING_PATH.unlink(missing_ok=True)
    _stage(
        "transactions",
        LEDGER_STAGED_PATH,
        mapping=mapping,
        window_days=window_days,
        active_days=window_days,
        label=f"Uploaded transactions ({filename})",
        currency=currency,
    )
    checks = [
        {"label": f"{report['rows_used']:,} transactions from {report['customers']:,} customers", "status": "pass"},
        {"label": f"History {report['first_date']} to {report['last_date']} ({report['span_days']} days)", "status": "pass"},
        {"label": f"Churn defined as: no purchase in the next {window_days} days", "status": "pass"},
    ]
    if report["rows_dropped_missing"]:
        checks.append({"label": f"{report['rows_dropped_missing']:,} rows dropped (missing id, date or amount)", "status": "warn"})
    if report["return_rows"]:
        checks.append({"label": f"{report['return_rows']:,} return/cancellation rows found and used", "status": "pass"})
    else:
        checks.append({"label": "No returns found in the file; return features will be empty", "status": "warn"})
    return render_upload_page(
        health_checks=checks,
        preview=build_preview_table(df),
        row_count=len(df),
        column_count=len(df.columns),
        uploaded_filename=filename,
        upload_success=True,
        saved_path=str(LEDGER_STAGED_PATH),
        ledger_summary=report | {"window_days": window_days},
    )


# ==================================================
# ONE-CLICK REAL-DATA DEMO
# ==================================================


@app.route("/use-real-demo", methods=["POST"])
@login_required
def use_real_demo():
    """Analyse the bundled real transaction history (UCI Online Retail)."""
    if not REAL_DEMO_PATH.exists():
        flash("The bundled real-data file was not found (data/external/).", "error")
        return redirect(url_for("upload_dataset"))
    columns = pd.read_csv(REAL_DEMO_PATH, nrows=100).columns
    _stage(
        "transactions",
        REAL_DEMO_PATH,
        mapping=transactions.detect_columns(columns),
        window_days=90,
        active_days=90,
        label="UCI Online Retail (real UK retailer transactions, 2010-11)",
        currency="\u00a3",
    )
    return _launch_analysis()


# ==================================================
# GENERATE A SYNTHETIC DATASET
# ==================================================


@app.route("/generate", methods=["POST"])
@login_required
def generate_dataset():
    """
    Create a synthetic customer dataset instead of an upload, using
    the customer count the user typed on the upload page.
    """
    customer_count = request.form.get("customer_count", "").strip()
    if not customer_count.isdigit() or int(customer_count) <= 0:
        flash("Please enter a valid number of customers (1 or more).", "error")
        return redirect(url_for("upload_dataset"))
    customer_count = int(customer_count)

    # Cap it so someone can't accidentally ask for millions of rows
    # and lock up the server. Lowered from 50,000 -> 5,000 for now:
    # every extra row adds real time to model training/clustering/SHAP,
    # and 5,000 is already far more than needed to see the full
    # dashboard in action. Raise this back up once you're not iterating
    # locally and want a bigger demo dataset.
    customer_count = min(customer_count, 5000)
    saved_path = UPLOAD_FOLDER / "customers_uploaded.csv"
    save_demo_dataset(customer_count, saved_path)
    _stage(
        "customers", saved_path, data_kind="synthetic", label="Synthetic demo data (not real customers)"
    )
    df = pd.read_csv(saved_path)
    flash(f"Generated {customer_count} synthetic customers.", "success")
    errors, warnings = validate_uploaded_data(df)
    return render_upload_page(
        preview=build_preview_table(df),
        row_count=len(df),
        column_count=len(df.columns),
        uploaded_filename=f"Generated ({customer_count} customers)",
        upload_success=True,
        saved_path=str(saved_path),
        health_checks=build_health_checks(df, errors, warnings),
    )


# ==================================================
# RUN CUSTORA ANALYSIS
# ==================================================


@app.route("/run-analysis", methods=["POST"])
@login_required
def run_analysis():
    return _launch_analysis()


def _launch_analysis():
    staged = load_staged()
    staged_path = Path(staged["path"]) if staged.get("path") else None
    if staged_path is not None and staged_path.exists():
        uploaded_path = staged_path
    else:
        uploaded_path = UPLOAD_FOLDER / "customers_uploaded.csv"
        staged = {"kind": "customers"}
    if not uploaded_path.exists():
        return render_upload_page(
            errors=[
                "No uploaded dataset was found.",
                "Please upload or generate a dataset first.",
            ],
        )

    # Start the pipeline on a background thread
    acquired = PIPELINE_LOCK.acquire(blocking=False)
    if acquired:
        print()
        print("=" * 70)
        print("CUSTORA WEB ANALYSIS STARTED (background thread)")
        print("=" * 70)
        _set_pipeline_state(status="running", run_id=None, error=None)

        # Reset the progress checklist to "not started" immediately, so
        # the status page's first poll doesn't briefly show the
        # *previous* run's (stale but still-OK) results before this
        # run's first stage actually finishes.
        save_pipeline_status([], PIPELINE_STAGES)
        thread = threading.Thread(
            target=_run_pipeline_in_background,
            args=(uploaded_path, staged),
            daemon=True,
        )
        thread.start()
    else:
        # A run is already in progress -- don't start a second one on
        # top of it, just send the user to watch the one already running.
        flash("An analysis is already running.", "warning")
    return redirect(url_for("pipeline_status_page"))


@app.route("/run-analysis/status")
@login_required
def pipeline_status_page():
    """Polling progress page shown while the background pipeline thread runs."""
    return render_template("pipeline_status.html", initial_stage_names=STAGE_NAMES)


@app.route("/run-analysis/status.json")
@login_required
def pipeline_status_json():
    """
    Polled by pipeline_status.html. Combines the in-memory run state
    (running/completed/failed -- authoritative, always current) with
    the on-disk per-stage checklist (reports/pipeline_status.csv,
    written after every stage -- may lag by a fraction of a second
    right as a new run starts and swaps in its own copy of that file,
    which is fine for a progress display).
    """
    state = _get_pipeline_state()
    stages = []
    if PIPELINE_STATUS_PATH.exists():
        try:
            status_df = pd.read_csv(PIPELINE_STATUS_PATH)
            # Blank CSV cells (Seconds/Error on SKIPPED rows) come back
            # from read_csv as float NaN. Python's json module -- what
            # jsonify uses -- serializes NaN as the bare token `NaN`,
            # which isn't valid JSON: browsers' JSON.parse rejects the
            # whole response with a SyntaxError the moment one shows up,
            # which is exactly what silently broke every poll once a
            # SKIPPED/blank row appeared. Swap NaN for None so it comes
            # out as JSON `null` instead.
            status_df = status_df.astype(object).where(pd.notnull(status_df), None)
            stages = status_df.to_dict("records")
        except Exception:
            stages = []

    status = state["status"]
    error = state["error"]

    # Fallbacks so the progress page can always reach a terminal state
    # and stop polling, even if the in-memory state is out of date --
    # e.g. the server was restarted mid-run (status resets to "idle"
    # while a finished checklist is still on disk), or the background
    # thread died before it could record its own outcome. Without this
    # the page polled "running"/"idle" forever with no way out.
    if stages:
        stage_statuses = [str(row.get("Status", "")) for row in stages]
        if status in ("idle", "running") and all(s == "OK" for s in stage_statuses):
            status = "completed"
        elif status in ("idle", "running") and any(s == "FAILED" for s in stage_statuses):
            status = "failed"
    elif status == "idle":
        # Nothing running and no checklist at all: don't leave the page
        # spinning on a run that isn't happening.
        status = "failed"
        error = error or "No analysis is currently running. Start one from the upload page."

    if status == "failed" and not error:
        failed = [str(row.get("Error", "")) for row in stages if row.get("Status") == "FAILED"]
        error = next((message for message in failed if message), "The pipeline did not finish.")

    response = jsonify(
        {
            "status": status,
            "run_id": state["run_id"],
            "error": error,
            "stages": stages,
            "dashboard_url": url_for("dashboard"),
            "upload_url": url_for("upload_dataset"),
        }
    )
    # Progress data is by definition stale the moment it's sent -- never
    # let a browser or proxy replay an old "running" response, which
    # would leave the page polling a run that already finished.
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/run-analysis/stage.json")
@login_required
def pipeline_stage_detail():
    """What one finished stage did to the user's data (read from the files that run left behind)."""
    name = request.args.get("name", "")
    if name not in STAGE_NAMES:
        abort(404)
    currency = (_read_json(RUN_META_FILE) or {}).get("currency") or "\u20b9"
    response = jsonify(stage_report.describe_stage(name, app.root_path, currency))
    response.headers["Cache-Control"] = "no-store"
    return response


# ==================================================
# ERROR HANDLERS
# ==================================================


@app.errorhandler(413)
def file_too_large(error):
    """
    Friendly response when an upload exceeds MAX_CONTENT_LENGTH,
    instead of the connection just hanging or a raw 413 page.
    """
    flash("That file is too large. The upload limit is 100 MB.", "error")
    return redirect(url_for("upload_dataset")), 413


# ==================================================
# RUN FLASK
# ==================================================

if __name__ == "__main__":

    # debug=True must never be used outside a trusted local machine --
    # it exposes the Werkzeug debugger, which allows arbitrary code
    # execution from the browser. Set FLASK_DEBUG=1 in the environment
    # to enable it for local development only.
    # threaded=True lets the dev server keep serving the progress-page
    # polling requests while a pipeline run's background thread is busy.
    app.run(debug=FLASK_DEBUG, threaded=True)
