"""
Per-run bookkeeping for the CuStorA pipeline.

WINDOWS COMPATIBILITY NOTE (why this file is simpler than it used to
be): every pipeline stage module (src/data/cleaning.py,
src/clustering/kmeans.py, etc.) reads and writes a fixed set of
relative paths -- "data/processed/X.csv", "reports/churn",
"models/churn", and so on. An earlier version of this file gave each
run its own directory under data/runs/<run_id>/ and used OS symlinks
(os.symlink) to atomically repoint those fixed paths at it, so two
overlapping runs couldn't clobber each other's output.

That symlink swap requires elevated privileges (or Developer Mode) to
create on Windows -- os.symlink raises a permission error without
either -- which made the app hang/fail silently on a stock Windows
setup. Since this is a single-user local app that already only allows
one run in flight at a time (see PIPELINE_LOCK in app.py), the
isolation the symlinks provided wasn't buying anything worth that
cost, so it's been removed: every stage module now reads and writes
those fixed paths directly, no symlink layer in between, on every OS.

What's kept from before: seeding data/raw/customers_messy.csv from
whatever was just uploaded/generated, and backing up the previous
run's raw dataset to data/raw/customers_previous.csv, same as always.
"""

import shutil
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

RUNS_DIR = PROJECT_ROOT / "data" / "runs"
CURRENT_RUN_FILE = RUNS_DIR / "current_run.txt"


def _new_run_id():
    """
    Sortable, collision-resistant run ID: UTC timestamp down to the
    millisecond. Used purely for logging/status display now (not for
    naming a per-run directory any more).
    """
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + f"_{int(time.time() * 1000) % 1000:03d}"


def start_new_run(uploaded_dataset_path):
    """
    Seed data/raw/customers_messy.csv from the just-uploaded/generated
    dataset (backing up whatever the previous run's raw dataset was to
    data/raw/customers_previous.csv first), and return a run_id used
    for logging/status display.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = _new_run_id()

    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    current_messy = raw_dir / "customers_messy.csv"
    if current_messy.exists():
        try:
            shutil.copy2(current_messy, raw_dir / "customers_previous.csv")
        except OSError as error:
            print("Warning: could not back up previous dataset:", error)

    shutil.copy2(uploaded_dataset_path, current_messy)
    CURRENT_RUN_FILE.write_text(run_id)
    return run_id
