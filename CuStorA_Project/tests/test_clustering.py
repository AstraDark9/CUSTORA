import pandas as pd

from src.clustering.kmeans import choose_k
from src.clustering.segment_naming import build_cluster_name, assign_segment_names


def test_choose_k_picks_the_highest_silhouette_row():
    results = pd.DataFrame(
        [
            {"K": 2, "Silhouette": 0.30, "DaviesBouldin": 1.1},
            {"K": 3, "Silhouette": 0.55, "DaviesBouldin": 0.9},  # best silhouette
            {"K": 4, "Silhouette": 0.40, "DaviesBouldin": 0.6},  # best DB, but not chosen
        ]
    )
    assert choose_k(results) == 3


def test_cluster_naming_reflects_high_value_dormant_pattern():
    # High monetary value, but the cluster hasn't purchased recently
    # (Recency normalized score is 0 = fully "bad"/dormant). Frequency
    # and engagement are left "average" so they don't crowd out the
    # two-descriptor name cap.
    scores = pd.Series({"Monetary": 1.0, "Frequency": 0.5, "EngagementScore": 0.5, "Recency": 0.0})
    name = build_cluster_name(scores)
    assert "High-Value" in name
    assert "Dormant" in name


def test_cluster_naming_caps_descriptors_at_two_for_readability():
    # A cluster that's high on every dimension shouldn't produce an
    # unreadable four-descriptor name -- only the two strongest show.
    scores = pd.Series({"Monetary": 1.0, "Frequency": 1.0, "EngagementScore": 1.0, "Recency": 1.0})
    name = build_cluster_name(scores)
    assert name.count(",") <= 1


def test_average_cluster_falls_back_to_core_customers():
    scores = pd.Series({"Monetary": 0.5, "Frequency": 0.5, "EngagementScore": 0.5, "Recency": 0.5})
    assert build_cluster_name(scores) == "Core Customers"


def test_segment_names_do_not_collide_across_distinct_clusters():
    data = pd.DataFrame(
        {
            "Cluster": [0, 0, 1, 1, 2, 2],
            "Monetary": [1000, 1100, 100, 120, 500, 520],
            "Frequency": [20, 22, 2, 3, 10, 11],
            "EngagementScore": [90, 92, 10, 12, 50, 52],
            "Recency": [2, 3, 300, 310, 60, 65],
        }
    )
    named = assign_segment_names(data)
    # Every distinct cluster should keep a unique segment name.
    mapping = named[["Cluster", "SegmentName"]].drop_duplicates()
    assert mapping["SegmentName"].is_unique
