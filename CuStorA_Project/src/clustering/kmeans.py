import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Row count above which evaluate_k_values scores Silhouette on a
# sample instead of the full dataset (see evaluate_k_values docstring
# -- Silhouette is O(n^2), so this is what keeps K selection's cost
# bounded on larger uploads/generated datasets).
SILHOUETTE_SAMPLE_THRESHOLD = 5000
SILHOUETTE_SAMPLE_SIZE = 2000

INPUT_FILE = "data/processed/customers_rfm.csv"
OUTPUT_FILE = "data/processed/customers_clustered.csv"


# Features used for customer segmentation.
#
# NOTE ON FEATURE OVERLAP: Complaints/Returns and ComplaintRate/
# ReturnRate measure the same underlying idea (service friction) --
# one as a raw count, one normalized by purchase volume. Including
# both would let "friction" quietly outweigh the other concepts in
# the distance calculation just because it's represented twice. We
# keep the *rate* versions only, since they're comparable across
# customers with very different purchase counts, and drop the raw
# counts here (they're still available elsewhere, e.g. in the churn
# model and on the customer profile page).
CLUSTER_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "EngagementScore",
    "ComplaintRate",
    "ReturnRate",
    "DiscountUsage",
    "TenureYears",
]


def prepare_data(data):
    """
    Prepare numerical customer features
    for K-Means clustering.
    """
    available_features = [feature for feature in CLUSTER_FEATURES if feature in data.columns]
    X = data[available_features].copy()
    return X, available_features


def reduce_outlier_influence(X):
    """
    Reduce the influence of extreme values
    without deleting the original customers.

    The original dataset remains unchanged.
    """
    X = X.copy()
    for column in X.columns:
        lower_limit = X[column].quantile(0.01)
        upper_limit = X[column].quantile(0.99)
        X[column] = X[column].clip(lower=lower_limit, upper=upper_limit)
    return X


def evaluate_k_values(X, min_k=2, max_k=8, sample_size=None):
    """
    Evaluate different K values using:

    - Silhouette Score
    - Davies-Bouldin Index

    silhouette_score computes a full pairwise distance matrix over
    its input, so it's O(n^2) in the number of rows -- fine at a few
    thousand rows, but it's what made K selection blow up on larger
    datasets (evaluated separately for every candidate K, so the cost
    multiplies by (max_k - min_k + 1) on top of that). Above
    SILHOUETTE_SAMPLE_THRESHOLD rows we instead score a random sample
    of SILHOUETTE_SAMPLE_SIZE points (silhouette_score's own
    sample_size argument), which turns each evaluation from O(n^2)
    into O(sample_size^2) -- a fixed cost regardless of dataset size.
    A few thousand-point sample is standard practice for silhouette
    on large datasets and gives a stable enough estimate to compare
    candidate Ks; Davies-Bouldin is left on the full dataset since
    it's computed from cluster centroids (O(n) to assign points, then
    O(k^2) between centroids) and isn't the bottleneck.
    """
    if sample_size is None:
        sample_size = SILHOUETTE_SAMPLE_SIZE if len(X) > SILHOUETTE_SAMPLE_THRESHOLD else None
    results = []
    print("\nEvaluating different K values...")
    if sample_size is not None:
        print(
            f"Dataset has {len(X)} rows (> {SILHOUETTE_SAMPLE_THRESHOLD}); "
            f"scoring Silhouette on a random sample of {sample_size} points per K."
        )
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        silhouette = silhouette_score(X, labels, sample_size=sample_size, random_state=42)
        davies_bouldin = davies_bouldin_score(X, labels)
        results.append({"K": k, "Silhouette": silhouette, "DaviesBouldin": davies_bouldin})
        print(f"K={k} | " f"Silhouette={silhouette:.4f} | " f"Davies-Bouldin={davies_bouldin:.4f}")
    return pd.DataFrame(results)


def choose_k(results):
    """
    Select the candidate K with the highest Silhouette Score.

    Silhouette is the *primary* (and only) selection criterion.
    Davies-Bouldin is computed and reported alongside it purely as a
    secondary diagnostic -- a sanity check the chosen K can be
    verified against, not a second vote in the decision. If the two
    metrics disagree sharply for the chosen K, that's a signal to look
    at the cluster profiles more closely, not a bug.
    """
    best_row = results.loc[results["Silhouette"].idxmax()]
    best_k = int(best_row["K"])
    return best_k


def create_final_clusters(X, data, best_k):
    """
    Perform final K-Means clustering.
    """
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    clustered_data = data.copy()
    clustered_data["Cluster"] = labels
    return clustered_data, model


def save_k_evaluation_chart(results, y_column, ylabel, title, output_path):
    """
    Shared plotting helper for the Silhouette and Davies-Bouldin
    charts -- same shape, just a different metric on the Y axis.
    """
    plt.figure(figsize=(9, 6))
    plt.plot(results["K"], results[y_column], marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(results["K"])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_cluster_profiles(data):
    """
    Create average behavioral profiles
    for each cluster.
    """
    profile_columns = [
        "Recency",
        "Frequency",
        "Monetary",
        "EngagementScore",
        "Complaints",
        "Returns",
        "DiscountUsage",
        "TenureYears",
    ]
    available_columns = [column for column in profile_columns if column in data.columns]
    profile = data.groupby("Cluster")[available_columns].mean()
    return profile


def main():
    print("\n" + "=" * 60)
    print("             CUSTORA K-MEANS SEGMENTATION")
    print("=" * 60)
    run()
    print("\nK-Means analysis completed successfully.")
    print("=" * 60)


def run(data=None, save=True):
    """
    Cluster customers by behavior and (by default) still write the
    same checkpoints as before: kmeans_evaluation.csv, cluster_profiles.csv,
    customers_clustered.csv, and the two evaluation charts. `data=None`
    reads the RFM dataset from disk for standalone/CLI use; pipeline.py
    instead passes the RFM DataFrame directly, in memory.
    """
    # Load data

    if data is None:
        data = pd.read_csv(INPUT_FILE)
    print(f"Customers loaded: {len(data)}")

    # Prepare features

    X, features = prepare_data(data)
    print("\nFeatures used for clustering:")
    for feature in features:
        print(f"  - {feature}")

    # Handle missing values

    print("\nHandling missing values...")
    X = X.fillna(X.median())

    # Reduce extreme outlier influence

    print("Reducing extreme outlier influence...")
    X = reduce_outlier_influence(X)

    # Standardize features

    print("Standardizing clustering features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Evaluate K values

    results = evaluate_k_values(X_scaled, min_k=3, max_k=6)

    # Save evaluation results

    if save:
        results.to_csv("reports/kmeans_evaluation.csv", index=False)
        print("\nK-Means evaluation saved to:" "\nreports/kmeans_evaluation.csv")

    # Choose candidate K

    best_k = choose_k(results)
    best_silhouette = results.loc[results["K"] == best_k, "Silhouette"].iloc[0]
    best_db = results.loc[results["K"] == best_k, "DaviesBouldin"].iloc[0]
    print(f"\nCandidate best K: {best_k}")
    print(f"Silhouette Score: " f"{best_silhouette:.4f}")
    print(f"Davies-Bouldin Index: " f"{best_db:.4f}")

    # Final clustering

    clustered_data, model = create_final_clusters(X_scaled, data, best_k)

    # Cluster sizes

    print("\nCluster Sizes:")
    cluster_sizes = clustered_data["Cluster"].value_counts().sort_index()
    for cluster, count in cluster_sizes.items():
        percentage = (count / len(clustered_data)) * 100
        print(f"Cluster {cluster}: " f"{count} customers " f"({percentage:.2f}%)")

    # Cluster profiles

    profile = create_cluster_profiles(clustered_data)
    print("\nCluster Profiles:")
    print(profile.to_string())

    if save:
        # Save profiles
        profile.to_csv("reports/cluster_profiles.csv")
        print("\nCluster profiles saved to:" "\nreports/cluster_profiles.csv")

        # Save clustered dataset
        clustered_data.to_csv(OUTPUT_FILE, index=False)
        print("\nClustered dataset saved to:" f"\n{OUTPUT_FILE}")

        # Create evaluation charts
        save_k_evaluation_chart(
            results,
            "Silhouette",
            "Silhouette Score",
            "CuStorA K-Means: Silhouette Score",
            "reports/silhouette_scores.png",
        )
        save_k_evaluation_chart(
            results,
            "DaviesBouldin",
            "Davies-Bouldin Index",
            "CuStorA K-Means: Davies-Bouldin Index",
            "reports/davies_bouldin_scores.png",
        )

    return clustered_data


if __name__ == "__main__":
    main()
