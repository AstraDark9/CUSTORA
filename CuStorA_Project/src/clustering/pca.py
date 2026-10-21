import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

INPUT_FILE = "data/processed/customers_rfm.csv"


def prepare_features(data):
    """
    Select numerical behavioral features for PCA.
    """
    features = [
        "Recency",
        "Frequency",
        "Monetary",
        "EngagementScore",
        "Complaints",
        "Returns",
        "DiscountUsage",
        "ComplaintRate",
        "ReturnRate",
        "TenureYears",
    ]
    available_features = [column for column in features if column in data.columns]
    X = data[available_features].copy()
    return X, available_features


def perform_pca(X):

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA with two components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    return (X_pca, explained_variance, scaler, pca)


def add_pca_columns(data):
    """
    Run PCA on this dataset's behavioral features and return a copy
    with PC1/PC2 columns added, plus the explained-variance ratios
    (for the console summary / caller to report).

    Note: PC1/PC2 are a visualization-only side branch -- K-Means
    clustering (kmeans.py) clusters on the original RFM features
    directly, not on these components, so this stage's output isn't
    on the critical path to segmentation; it only feeds the PCA
    scatter plot.
    """
    X, features = prepare_features(data)
    X_pca, explained_variance, scaler, pca = perform_pca(X)
    data = data.copy()
    data["PC1"] = X_pca[:, 0]
    data["PC2"] = X_pca[:, 1]
    return data, features, explained_variance


def save_pca_plot(data, output_image="reports/pca_visualization.png"):
    plt.figure(figsize=(9, 6))
    plt.scatter(data["PC1"], data["PC2"], alpha=0.6)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Custora Customer PCA Visualization")
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"PCA visualization saved to:" f"\n{output_image}")


def run(data=None, save=True):
    """
    Add PCA coordinates for the visualization and (by default) still
    write customers_pca.csv + the scatter plot, same as before.
    `data=None` reads the RFM dataset from disk for standalone/CLI
    use; pipeline.py instead passes the RFM DataFrame directly.
    """
    if data is None:
        data = pd.read_csv(INPUT_FILE)
    data, features, explained_variance = add_pca_columns(data)
    print("\nFeatures used for PCA:")
    for feature in features:
        print(f"  - {feature}")
    print(f"\nPC1 explained variance: " f"{explained_variance[0] * 100:.2f}%")
    print(f"PC2 explained variance: " f"{explained_variance[1] * 100:.2f}%")
    print(f"Total explained variance: " f"{explained_variance.sum() * 100:.2f}%")
    if save:
        output_file = "data/processed/customers_pca.csv"
        data.to_csv(output_file, index=False)
        print(f"\nPCA dataset saved to:" f"\n{output_file}")
        save_pca_plot(data)
    return data


def main():
    print("\nLoading RFM dataset...")
    run()


if __name__ == "__main__":
    main()
