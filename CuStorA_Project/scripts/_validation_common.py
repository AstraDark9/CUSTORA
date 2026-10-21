"""
Shared mechanics for CuStorA's external-dataset validation scripts
(run_external_validation.py = Telco, run_external_validation_ecommerce.py
= E-Commerce). Both scripts run the same four steps against a different
real dataset; this module holds the step logic once so each script only
has to state what's different about its dataset (path, source, output
dir, native-feature prep, segmentation columns) plus its own README
writeup.
"""

import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.data.column_mapper import suggest_mapping, build_coverage_report, apply_mapping
from src.data.cleaning import clean_dataset
from src.features.engineering import create_features
from src.rfm.rfm import calculate_rfm_scores
from src.clustering.kmeans import prepare_data as kmeans_prepare_data
from src.churn import churn_models


def step_1_column_mapping(df, dataset_name, output_dir):
    print(f"\n{'='*60}\nSTEP 1: COLUMN MAPPING (honest, no manual overrides)\n{'='*60}")
    mapping_result = suggest_mapping(list(df.columns))
    coverage_report = build_coverage_report(mapping_result, dataset_name=dataset_name)
    print(coverage_report.to_string(index=False))
    os.makedirs(output_dir, exist_ok=True)
    coverage_report.to_csv(f"{output_dir}/column_mapping_coverage.csv", index=False)
    return mapping_result, coverage_report


def _try_stage(label, fn):
    try:
        return {"Stage": label, "Result": fn()}
    except Exception as e:
        return {"Stage": label, "Result": f"FAILED: {type(e).__name__}: {e}"}


def step_2_what_breaks(df, mapping_result, output_dir):
    """Push the honestly-mapped columns through CuStorA's real cleaning /
    feature-engineering / RFM / clustering-prep stages, each tried
    independently, and record the first failure at each stage."""
    print(f"\n{'='*60}\nSTEP 2: WHAT BREAKS (using only the honestly-mapped columns)\n{'='*60}")
    mapped_df = apply_mapping(df, mapping_result)

    def check_cluster_features():
        X, feats = kmeans_prepare_data(mapped_df)
        if not feats:
            raise ValueError("None of CLUSTER_FEATURES exist in the mapped data")
        return f"Succeeded, but only {len(feats)}/8 cluster features available: {feats}"

    failures = [
        _try_stage(
            "src/data/cleaning.py: clean_dataset()",
            lambda: (clean_dataset(mapped_df), "Succeeded")[1],
        ),
        _try_stage(
            "src/features/engineering.py: create_features()",
            lambda: (create_features(mapped_df), "Succeeded")[1],
        ),
        _try_stage(
            "src/rfm/rfm.py: calculate_rfm_scores()",
            lambda: (calculate_rfm_scores(mapped_df), "Succeeded")[1],
        ),
        _try_stage("src/clustering/kmeans.py: prepare_data()", check_cluster_features),
    ]

    failures_df = pd.DataFrame(failures)
    print(failures_df.to_string(index=False))
    failures_df.to_csv(f"{output_dir}/pipeline_stage_failures.csv", index=False)
    return failures_df


def step_3_native_churn_model(df, output_dir, native_filename, prepare_fn=None):
    """Run CuStorA's churn-modeling *methodology* (split/calibration/
    threshold selection) unmodified, directly on the dataset's own
    native columns -- not a CuStorA-schema reshaping of them. Possible
    because prepare_features()/build_preprocessor() auto-detect
    numeric vs. categorical columns rather than hardcoding CuStorA's
    feature names."""
    print(f"\n{'='*60}\nSTEP 3: CHURN METHODOLOGY ON NATIVE FEATURES\n{'='*60}")
    native = prepare_fn(df.copy()) if prepare_fn else df.copy()

    model_dir, report_dir = f"{output_dir}/churn_model", f"{output_dir}/churn_reports"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    native_path = f"{output_dir}/{native_filename}"
    native.to_csv(native_path, index=False)

    return churn_models.run(data_path=native_path, model_dir=model_dir, report_dir=report_dir)


def step_4_segmentation(df, features, output_dir, filename_prefix=""):
    """Small K-Means run on the dataset's own numeric fields, to see
    whether the "few, statistically-supported clusters" pattern from
    src/clustering/cluster_investigation.py holds on real data too."""
    print(f"\n{'='*60}\nSTEP 4: SEGMENTATION ON NATIVE NUMERIC FEATURES\n{'='*60}")
    X = df[features].copy()
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(X.median())
    X_scaled = StandardScaler().fit_transform(X)

    rows, profiles = [], {}
    for k in (2, 3, 4):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        rows.append({"K": k, "Silhouette": round(silhouette_score(X_scaled, labels), 4)})
        profiled = X.copy()
        profiled["Cluster"] = labels
        profiles[k] = profiled.groupby("Cluster").mean().round(2)

    comparison = pd.DataFrame(rows)
    print(comparison.to_string(index=False))
    comparison.to_csv(f"{output_dir}/{filename_prefix}segmentation_k_comparison.csv", index=False)

    best_k = int(comparison.loc[comparison["Silhouette"].idxmax(), "K"])
    profiles[best_k].to_csv(f"{output_dir}/{filename_prefix}segmentation_best_k_profile.csv")
    print(f"\nBest K by silhouette: {best_k}")
    print(profiles[best_k].to_string())
    return comparison, profiles, best_k


def run_validation(
    dataset_path,
    dataset_name,
    output_dir,
    native_filename,
    segmentation_features,
    write_readme_fn,
    native_prepare_fn=None,
    banner="",
    segmentation_prefix="",
):
    """Runs steps 1-4 against one external dataset and writes its README."""
    print(f"\n{'='*60}\n   {banner}\n{'='*60}")

    df = pd.read_csv(dataset_path)
    print(f"\nLoaded {len(df)} customers, {len(df.columns)} columns from:\n{dataset_path}")

    mapping_result, coverage_report = step_1_column_mapping(df, dataset_name, output_dir)
    failures_df = step_2_what_breaks(df, mapping_result, output_dir)
    churn_result = step_3_native_churn_model(df, output_dir, native_filename, native_prepare_fn)
    seg_comparison, seg_profiles, seg_best_k = step_4_segmentation(
        df, segmentation_features, output_dir, segmentation_prefix
    )

    write_readme_fn(
        df, coverage_report, failures_df, churn_result, seg_comparison, seg_profiles, seg_best_k
    )

    print(f"\n{'='*60}\nEXTERNAL VALIDATION COMPLETE -- see {output_dir}/README.md\n{'='*60}")
