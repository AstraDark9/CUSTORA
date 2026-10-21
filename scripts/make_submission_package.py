"""
Build a lean CuStorA submission/deployment package.

CuStorA's real-data demo and Model evidence page read
data/external/online_retail_transactions_uci.csv and reports/external_validation*,
so those are INCLUDED. Only scratch material is left out: per-run
workspaces, the clustering investigation, derived snapshot CSVs and caches.

Usage:
    python scripts/make_submission_package.py [output_zip_path]

Defaults to CuStorA_submission.zip in the project root.
"""

import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories to exclude entirely (research/dev-only, or generated and
# reproducible from a fresh `python app.py` + "Generate Dataset" + "Run
# Analysis"). Paths are relative to PROJECT_ROOT.
EXCLUDE_DIRS = {
    "reports/clustering_investigation",  # "why K=2" investigation, not app output
    "data/runs",  # per-run scratch workspace (gitignored; regenerated per run)
    "__pycache__",
    ".git",
    ".pytest_cache",
}

# Individual files to exclude (large, derived from the excluded research
# datasets above -- not needed to run the app).
EXCLUDE_FILES = {
    "data/processed/online_retail_temporal_snapshots.csv",  # ~2.4 MB, derived from UCI dataset
}

# Research scripts that only make sense with the excluded datasets above.
EXCLUDE_SCRIPTS = {
    "scripts/build_temporal_snapshots_online_retail.py",
    "scripts/run_temporal_validation_online_retail.py",
}

EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def _is_excluded(rel_path: Path) -> bool:
    rel_str = rel_path.as_posix()
    if rel_path.suffix in EXCLUDE_SUFFIXES:
        return True
    if rel_str in EXCLUDE_FILES or rel_str in EXCLUDE_SCRIPTS:
        return True
    for part in rel_path.parts:
        if part in EXCLUDE_DIRS:
            return True
    for excluded_dir in EXCLUDE_DIRS:
        if rel_str == excluded_dir or rel_str.startswith(excluded_dir + "/"):
            return True
    return False


def build_package(output_path: Path) -> tuple[int, int]:
    """Zip everything in PROJECT_ROOT except EXCLUDE_*. Returns (file_count, total_uncompressed_bytes)."""
    file_count = 0
    total_bytes = 0
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(PROJECT_ROOT.rglob("*")):
            if path.is_dir():
                continue
            rel_path = path.relative_to(PROJECT_ROOT)
            if output_path.resolve() == path.resolve():
                continue
            if _is_excluded(rel_path):
                continue
            zf.write(path, arcname=str(rel_path))
            file_count += 1
            total_bytes += path.stat().st_size
    return file_count, total_bytes


def main():
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "CuStorA_submission.zip"
    file_count, total_bytes = build_package(output_path)
    compressed_bytes = output_path.stat().st_size
    print(f"Wrote {output_path}")
    print(f"  {file_count} files, {total_bytes / 1_000_000:.1f} MB uncompressed")
    print(f"  {compressed_bytes / 1_000_000:.1f} MB compressed")


if __name__ == "__main__":
    main()
