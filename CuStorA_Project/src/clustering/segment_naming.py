"""
Turns numerical K-Means cluster IDs into business-friendly segment
names -- works for any number of clusters (K-Means picks K
automatically in kmeans.py, so this can't assume K=2).

A cluster is named from ITS OWN characteristics (is it high/low on
monetary value, frequency, recency, engagement?) rather than being
forced onto a single low-to-high "value ladder". A ladder assumes
value is one-dimensional, which breaks down the moment a cluster is
high on monetary value but low on engagement, or vice versa -- that
cluster isn't meaningfully "above" or "below" another one, it's just
different. Naming from characteristics avoids implying a value
ranking the data doesn't actually support.
"""

import pandas as pd

INPUT_FILE = "data/processed/customers_clustered.csv"
OUTPUT_FILE = "data/processed/customers_segmented.csv"

# Columns used to characterize each cluster, and whether a HIGH value
# on that column is the "good" direction. Recency is inverted because
# lower recency (fewer days since last purchase) is the good outcome.
PROFILE_COLUMNS = {
    "Monetary": True,
    "Frequency": True,
    "EngagementScore": True,
    "Recency": False,
}

# A cluster's normalized (0-1) score on a column is labeled HIGH above
# this and LOW below this; scores in between are considered "Medium"
# and left out of the name so it doesn't get overly long.
HIGH_CUTOFF = 0.62
LOW_CUTOFF = 0.38

# Short label used per column when a cluster scores HIGH/LOW on it,
# in priority order (first match per direction is used in the name).
LABELS = {
    "Monetary": {"High": "High-Value", "Low": "Low-Spend"},
    "Frequency": {"High": "Frequent", "Low": "Occasional"},
    "EngagementScore": {"High": "High-Engagement", "Low": "Low-Engagement"},
    "Recency": {"High": "Active", "Low": "Dormant"},  # High = good = low recency
}


def profile_clusters(data):
    """
    Each cluster's standing on every profiling column, as a 0-1 score.

    Score = the average PERCENTILE RANK (among all customers) of the
    cluster's members. Percentiles are used instead of rescaling raw
    means between the lowest and highest cluster because real spend data
    is heavily skewed: one small cluster of very large accounts would
    otherwise stretch the scale and make every other cluster look
    "low-spend". 0.5 means "typical customer"; near 1 / near 0 means the
    cluster's members are mostly in the top / bottom of the customer base.
    """
    columns = [c for c in PROFILE_COLUMNS if c in data.columns]
    ranks = data[columns].rank(pct=True)
    ranks["Cluster"] = data["Cluster"].values
    normalized = ranks.groupby("Cluster")[columns].mean()

    # Flip columns where a LOW raw value is the good outcome (Recency),
    # so "High" always means "good" in the normalized score.
    for column, higher_is_better in PROFILE_COLUMNS.items():
        if column in normalized.columns and not higher_is_better:
            normalized[column] = 1 - normalized[column]
    return normalized


def build_cluster_name(cluster_scores):
    """
    Build a short descriptive name from a cluster's normalized scores,
    e.g. "High-Value, Dormant" or "Frequent, Low-Engagement". Falls
    back to "Core Customers" if nothing is distinctly high or low
    (i.e. the cluster is unremarkable/average on every dimension).
    """
    descriptors = []
    for column in ["Monetary", "Frequency", "EngagementScore", "Recency"]:
        if column not in cluster_scores:
            continue
        score = cluster_scores[column]
        if score >= HIGH_CUTOFF:
            descriptors.append(LABELS[column]["High"])
        elif score <= LOW_CUTOFF:
            descriptors.append(LABELS[column]["Low"])
    if not descriptors:
        return "Core Customers"

    # Keep it readable: at most the two strongest descriptors.
    return ", ".join(descriptors[:2]) + " Customers"


def assign_segment_names(data):
    """
    Convert numerical K-Means cluster IDs into names built from each
    cluster's own characteristics (see build_cluster_name).
    """
    data = data.copy()
    profiles = profile_clusters(data)
    names_by_cluster = {
        cluster_id: build_cluster_name(profiles.loc[cluster_id]) for cluster_id in profiles.index
    }

    # If two clusters land on the exact same name (possible with
    # coarse High/Low buckets), disambiguate rather than silently
    # merging them on the dashboard.
    seen = {}
    for cluster_id, name in names_by_cluster.items():
        seen.setdefault(name, []).append(cluster_id)
    tiers = ["top tier", "core", "entry", "light"]
    for name, cluster_ids in seen.items():
        if len(cluster_ids) > 1:
            # Tell look-alike clusters apart by how much they spend (readable),
            # falling back to the cluster number if that is not available.
            try:
                ranked = sorted(cluster_ids, key=lambda c: -float(profiles.loc[c, "Monetary"]))
            except (KeyError, TypeError, ValueError):
                ranked = list(cluster_ids)
            for position, cluster_id in enumerate(ranked):
                tag = tiers[position] if position < len(tiers) and "Monetary" in profiles.columns else f"Segment {cluster_id}"
                names_by_cluster[cluster_id] = f"{name} ({tag})"
    data["SegmentName"] = data["Cluster"].map(names_by_cluster)
    return data


def run(data=None, save=True):
    """
    Name clusters and (by default) still write customers_segmented.csv,
    same as before. `data=None` reads the clustered dataset from disk
    for standalone/CLI use; pipeline.py instead passes the clustered
    DataFrame directly, in memory.
    """
    if data is None:
        data = pd.read_csv(INPUT_FILE)
    print(f"Customers loaded: {len(data)}")
    data = assign_segment_names(data)
    print("\nCustomer segments (low to high value):")
    print(
        data[["Cluster", "SegmentName"]]
        .drop_duplicates()
        .sort_values("Cluster")
        .to_string(index=False)
    )
    print("\nSegment distribution:")
    print(data["SegmentName"].value_counts())
    if save:
        data.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSegmented dataset saved to:" f"\n{OUTPUT_FILE}")
    return data


def main():
    print("\n" + "=" * 55)
    print("           CUSTORA SEGMENT NAMING")
    print("=" * 55)
    print("\nLoading clustered dataset...")
    run()
    print("\nSegment naming completed.")
    print("=" * 55)


if __name__ == "__main__":
    main()
