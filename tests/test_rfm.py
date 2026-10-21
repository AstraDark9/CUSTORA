import numpy as np
import pandas as pd

from src.rfm.rfm import calculate_rfm_scores, create_rfm_segments


def _feature_frame(n=50, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "CustomerID": [f"CUST{i}" for i in range(n)],
            "Recency": rng.integers(0, 400, n),
            "Frequency": rng.integers(1, 50, n),
            "Monetary": rng.uniform(10, 5000, n),
        }
    )


def test_rfm_scores_are_within_1_to_5():
    scored = calculate_rfm_scores(_feature_frame())
    for column in ["R_Score", "F_Score", "M_Score"]:
        assert scored[column].between(1, 5).all()


def test_lower_recency_gets_a_higher_r_score():
    data = _feature_frame()
    scored = calculate_rfm_scores(data)
    most_recent = scored.loc[scored["Recency"].idxmin()]
    least_recent = scored.loc[scored["Recency"].idxmax()]
    assert most_recent["R_Score"] >= least_recent["R_Score"]


def test_rfm_total_is_sum_of_the_three_scores():
    scored = calculate_rfm_scores(_feature_frame())
    expected = scored["R_Score"] + scored["F_Score"] + scored["M_Score"]
    assert (scored["RFM_Total"] == expected).all()


def test_rfm_segments_are_from_the_known_set():
    scored = calculate_rfm_scores(_feature_frame())
    segmented = create_rfm_segments(scored)
    valid_segments = {
        "Champions",
        "Loyal Customers",
        "High Value",
        "At Risk",
        "Needs Attention",
        "Regular Customers",
    }
    assert set(segmented["RFM_Segment"].unique()).issubset(valid_segments)
