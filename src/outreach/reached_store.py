"""
Remembers which customers your team has already reached out to.

Why a small JSON file: this is a single-user local app, the list is short, and a
plain file survives restarts without adding a database. Writes are atomic (write a
temp file, then replace) so a crash mid-save can never leave a half-written list.

The list belongs to ONE dataset. It stores a fingerprint of the dataset it was
made for; if a different dataset is analysed later, the old list is ignored
instead of hiding unrelated customers who happen to share an ID.
"""

import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path

STORE_PATH = Path(__file__).resolve().parents[2] / "data" / "reached_customers.json"
_LOCK = threading.Lock()

# What we remember about a customer at the moment they were marked as reached,
# so the Reached page shows the situation your team acted on.
SNAPSHOT_FIELDS = (
    "PredictedChurnProbability",
    "RevenueAtStake",
    "ExpectedLoss",
    "CustomerState",
    "SegmentName",
    "Recommendation",
    "BusinessPriority",
)


def _read(path):
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load(fingerprint, path=None):
    """{customer_id: record} for this dataset (empty if the list belongs to another dataset)."""
    data = _read(path or STORE_PATH)
    if data.get("dataset") != fingerprint:
        return {}
    customers = data.get("customers")
    return customers if isinstance(customers, dict) else {}


def reached_ids(fingerprint, path=None):
    return set(load(fingerprint, path))


def mark(customer_id, snapshot, fingerprint, path=None):
    """Record that a customer was reached. Returns the stored record."""
    path = path or STORE_PATH
    record = {
        "reached_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "snapshot": {k: snapshot.get(k) for k in SNAPSHOT_FIELDS},
    }
    with _LOCK:
        customers = dict(load(fingerprint, path))
        customers[str(customer_id)] = record
        _write(path, {"dataset": fingerprint, "customers": customers})
    return record


def unmark(customer_id, fingerprint, path=None):
    """Move a customer back to the action queue. Returns True if they were on the list."""
    path = path or STORE_PATH
    with _LOCK:
        customers = dict(load(fingerprint, path))
        existed = customers.pop(str(customer_id), None) is not None
        if existed:
            _write(path, {"dataset": fingerprint, "customers": customers})
    return existed
