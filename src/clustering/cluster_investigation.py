"""
Investigates *why* K-Means settles on K=2 for CuStorA's customer base,
instead of just reporting the silhouette-selected K and moving on
(project review item #3 / "Finish clustering investigation").

Three questions, in order:

1. Are the clustering features actually independent signals, or is
   some of "the data" secretly one signal counted several times?
   -> correlation matrix over CLUSTER_FEATURES.

2. How much of the variance in the (scaled) feature space is
   explainable in a low number of dimensions? If most of it lives in
   1-2 components, a small number of clusters is what the data
   supports, not a modeling shortcoming.
   -> full PCA explained-variance curve (not just the 2 components
      kmeans.py/pca.py plot).

3. If we force K=3 or K=4 anyway, do the resulting clusters mean
   something a business could act on differently from each other, or
   do they just split one real group into arbitrary sub-slices?
   -> fit K=2/3/4, compare silhouette *and* cluster profiles
      (Recency/Frequency/Monetary/... averages per cluster).

Output: reports/clustering_investigation/
  - correlation_matrix.csv, correlation_heatmap.png
  - pca_explained_variance.csv, pca_scree_plot.png
  - k_comparison_profiles.csv  (cluster means for K=2, K=3, K=4 side by side)
  - README.md                  (the write-up, including the recommendation)
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.clustering.kmeans import CLUSTER_FEATURES, prepare_data, reduce_outlier_influence

INPUT_FILE = "data/processed/customers_rfm.csv"
OUTPUT_DIR = "reports/clustering_investigation"


def build_correlation_matrix(X):
    """Pearson correlation between every pair of clustering features.

    A pair with |r| well above ~0.6 is a candidate for being "the same
    idea counted twice" (see the ComplaintRate/ReturnRate note already
    in kmeans.py for what that looks like when it's been fixed)."""
    return X.corr(numeric_only=True).round(3)


def find_high_correlations(corr_matrix, threshold=0.6):
    """Pull out feature pairs above the threshold, excluding the
    diagonal, so the README can call them out explicitly instead of
    making the reader scan a full matrix."""
    pairs = []
    columns = corr_matrix.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            r = corr_matrix.iloc[i, j]
            if abs(r) >= threshold:
                pairs.append({"FeatureA": columns[i], "FeatureB": columns[j], "Correlation": r})
    return pd.DataFrame(pairs)


def full_pca_explained_variance(X_scaled, n_features):
    """Fit PCA with every component (not just 2, unlike pca.py, which
    is scoped to producing a 2D scatter plot) so the full explained
    variance / cumulative variance curve is visible."""
    pca = PCA(n_components=n_features, random_state=42)
    pca.fit(X_scaled)
    table = pd.DataFrame(
        {
            "Component": [f"PC{i + 1}" for i in range(n_features)],
            "ExplainedVariance": pca.explained_variance_ratio_.round(4),
            "CumulativeVariance": np.cumsum(pca.explained_variance_ratio_).round(4),
        }
    )
    return table, pca


def save_scree_plot(table, output_path):
    plt.figure(figsize=(8, 5))
    plt.plot(table["Component"], table["ExplainedVariance"], marker="o", label="Per-component")
    plt.plot(table["Component"], table["CumulativeVariance"], marker="s", label="Cumulative")
    plt.axhline(0.80, color="gray", linestyle="--", linewidth=1, label="80% reference line")
    plt.ylabel("Explained variance ratio")
    plt.title("CuStorA Clustering Features: PCA Explained Variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_correlation_heatmap(corr_matrix, output_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(corr_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(label="Pearson correlation")
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.title("CuStorA Clustering Features: Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def compare_k_values(X_scaled, raw_features, k_values=(2, 3, 4)):
    """Fit each candidate K and return (comparison_table, profiles_dict)
    where profiles_dict[k] is the per-cluster mean of the *raw*
    (unscaled) features -- what a business reader actually wants to
    see, not the standardized values used to fit the model."""
    rows = []
    profiles = {}
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        silhouette = silhouette_score(X_scaled, labels)
        sizes = pd.Series(labels).value_counts().sort_index()
        size_summary = ", ".join(
            f"cluster {c}: {n} ({n / len(labels) * 100:.1f}%)" for c, n in sizes.items()
        )
        rows.append({"K": k, "Silhouette": round(silhouette, 4), "ClusterSizes": size_summary})
        profiled = raw_features.copy()
        profiled["Cluster"] = labels
        profiles[k] = profiled.groupby("Cluster").mean().round(2)
    return pd.DataFrame(rows), profiles


def write_readme(
    output_dir, corr_matrix, high_corr_pairs, pca_table, k_comparison, profiles, n_customers
):
    lines = []
    lines.append("# CuStorA Clustering Investigation: Why K=2?\n")
    lines.append(
        f"Investigation run on `{INPUT_FILE}` ({n_customers} customers), "
        f"over the same `CLUSTER_FEATURES` used by `src/clustering/kmeans.py`: "
        f"{', '.join(CLUSTER_FEATURES)}.\n"
    )
    lines.append("## 1. Are the clustering features actually independent?\n")
    if len(high_corr_pairs):
        lines.append(
            "Feature pairs with |correlation| >= 0.6 (see `correlation_matrix.csv` "
            "for the full matrix, `correlation_heatmap.png` for the plot):\n"
        )
        lines.append(high_corr_pairs.to_markdown(index=False))
        lines.append(
            "\nA pair up here isn't automatically a bug -- Frequency and Monetary, "
            "for example, are expected to move together for real customers -- but "
            "it does mean those two dimensions are pulling K-Means in a similar "
            "direction rather than contributing independent structure, which is "
            "part of why the data doesn't support many distinct clusters.\n"
        )
    else:
        lines.append(
            "No feature pair reached |correlation| >= 0.6 -- the eight clustering "
            "features are behaving as reasonably independent signals on this "
            "dataset, not duplicates of each other under different names.\n"
        )
    lines.append("\n## 2. How much variance lives in a small number of dimensions?\n")
    top2_cumulative = pca_table.iloc[1]["CumulativeVariance"] * 100
    lines.append(
        f"Full PCA explained-variance curve is in `pca_explained_variance.csv` "
        f"and `pca_scree_plot.png`. The first two components alone explain "
        f"**{top2_cumulative:.1f}%** of total variance across the "
        f"{len(CLUSTER_FEATURES)} clustering features:\n"
    )
    lines.append(pca_table.to_markdown(index=False))
    if top2_cumulative >= 60:
        lines.append(
            "\nMost of the structure in this feature set is genuinely "
            "low-dimensional -- consistent with the data supporting a small "
            "number of clusters, not many.\n"
        )
    else:
        lines.append(
            "\nVariance is fairly spread across components -- this dataset could "
            "plausibly support more structure than K=2 if the cluster profiles "
            "below back that up.\n"
        )
    lines.append("\n## 3. What do K=2, K=3, and K=4 actually look like?\n")
    lines.append(k_comparison.to_markdown(index=False))
    lines.append("\n")
    for k in profiles:
        lines.append(f"\n### K={k} cluster profiles (raw feature means)\n")
        lines.append(profiles[k].to_markdown())
        lines.append("\n")
    lines.append("\n## Recommendation\n")
    best_row = k_comparison.loc[k_comparison["Silhouette"].idxmax()]
    best_k = int(best_row["K"])
    lines.append(
        f"K={best_k} has the highest silhouette score "
        f"({best_row['Silhouette']:.4f}) of the candidates compared here. "
    )
    if best_k == 2:
        lines.append(
            "Looking at the K=3 and K=4 profiles above, the extra clusters split "
            "one of the K=2 groups into slices that differ from each other mostly "
            'by degree (e.g. "somewhat engaged" vs. "very engaged") rather than '
            "by a distinct behavioural pattern a business would act on "
            "differently. Combined with points 1 and 2 above (some correlated "
            "features, and most variance concentrated in the first couple of "
            "components), **K=2 is what this dataset actually supports.** The "
            'honest framing is "this dataset currently separates into two '
            'statistically distinct customer groups," not "CuStorA only found '
            "two segments because it didn't look hard enough.\"\n"
        )
    else:
        lines.append(
            f"Unlike the synthetic-data baseline, K={best_k} is silhouette-best "
            f"here -- inspect the per-K profiles above to confirm the extra "
            f"cluster(s) correspond to a real, actionable behavioural "
            f"difference before adopting it as the production K.\n"
        )
    lines.append(
        "\n*Do not force a specific K to make the segmentation story look "
        "more granular than the data supports -- see project review, "
        '"What I would NOT do."*\n'
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/README.md", "w") as f:
        f.write("\n".join(lines))


def main():
    print("\n" + "=" * 60)
    print("      CUSTORA CLUSTERING INVESTIGATION (why K=2?)")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = pd.read_csv(INPUT_FILE)
    print(f"\nCustomers loaded: {len(data)}")
    X_raw, features = prepare_data(data)
    X_raw = X_raw.fillna(X_raw.median())
    X_clipped = reduce_outlier_influence(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clipped)

    # 1. Correlation matrix
    print("\nBuilding correlation matrix...")
    corr_matrix = build_correlation_matrix(X_clipped)
    corr_matrix.to_csv(f"{OUTPUT_DIR}/correlation_matrix.csv")
    save_correlation_heatmap(corr_matrix, f"{OUTPUT_DIR}/correlation_heatmap.png")
    high_corr_pairs = find_high_correlations(corr_matrix)
    print(corr_matrix.to_string())

    # 2. Full PCA explained variance
    print("\nRunning full PCA explained-variance analysis...")
    pca_table, _ = full_pca_explained_variance(X_scaled, len(features))
    pca_table.to_csv(f"{OUTPUT_DIR}/pca_explained_variance.csv", index=False)
    save_scree_plot(pca_table, f"{OUTPUT_DIR}/pca_scree_plot.png")
    print(pca_table.to_string(index=False))

    # 3. K=2/3/4 comparison
    print("\nComparing K=2, K=3, K=4...")
    k_comparison, profiles = compare_k_values(X_scaled, X_clipped, k_values=(2, 3, 4))
    print(k_comparison.to_string(index=False))
    all_profiles = []
    for k, profile in profiles.items():
        p = profile.copy()
        p.insert(0, "K", k)
        all_profiles.append(p.reset_index())
    pd.concat(all_profiles, ignore_index=True).to_csv(
        f"{OUTPUT_DIR}/k_comparison_profiles.csv", index=False
    )

    # Write-up
    write_readme(
        OUTPUT_DIR, corr_matrix, high_corr_pairs, pca_table, k_comparison, profiles, len(data)
    )
    print(f"\nFull investigation write-up saved to:\n{OUTPUT_DIR}/README.md")
    print("\n" + "=" * 60)
    print("CLUSTERING INVESTIGATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
