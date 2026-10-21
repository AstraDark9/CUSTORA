import pandas as pd

INPUT_FILE = "data/processed/customers_features.csv"
OUTPUT_FILE = "data/processed/customers_rfm.csv"


def calculate_rfm_scores(data):
    """
    Calculate RFM scores for each customer.
    """
    data = data.copy()

    # RECENCY SCORE
    # Lower recency is better.

    data["R_Score"] = pd.qcut(data["Recency"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop")

    # FREQUENCY SCORE
    # Higher frequency is better.

    data["F_Score"] = pd.qcut(
        data["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
    )

    # MONETARY SCORE
    # Higher monetary value is better.

    data["M_Score"] = pd.qcut(
        data["Monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
    )

    # Convert scores to integers
    data["R_Score"] = data["R_Score"].astype(int)
    data["F_Score"] = data["F_Score"].astype(int)
    data["M_Score"] = data["M_Score"].astype(int)

    # Combined RFM score
    data["RFM_Score"] = (
        data["R_Score"].astype(str) + data["F_Score"].astype(str) + data["M_Score"].astype(str)
    )

    # Numerical RFM total
    data["RFM_Total"] = data["R_Score"] + data["F_Score"] + data["M_Score"]
    return data


def create_rfm_segments(data):
    """
    Create interpretable RFM customer segments.
    """

    def assign_segment(row):
        r = row["R_Score"]
        f = row["F_Score"]
        m = row["M_Score"]

        # High-value recent and frequent customers
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"

        # Recent customers with good behavior
        elif r >= 4 and f >= 3:
            return "Loyal Customers"

        # High monetary value
        elif m >= 4:
            return "High Value"

        # Customers who haven't purchased recently
        elif r <= 2 and f >= 3:
            return "At Risk"

        # Low engagement / low purchase activity
        elif r <= 2 and f <= 2:
            return "Needs Attention"
        else:
            return "Regular Customers"

    data["RFM_Segment"] = data.apply(assign_segment, axis=1)
    return data


def run(data=None, save=True):
    """
    Score and segment customers by RFM, writing customers_rfm.csv by
    default (same checkpoint as before). `data=None` reads the feature
    dataset from disk for standalone/CLI use; pipeline.py instead
    passes the feature-engineered DataFrame directly, in memory.
    """
    if data is None:
        data = pd.read_csv(INPUT_FILE)
    data = calculate_rfm_scores(data)
    data = create_rfm_segments(data)
    if save:
        data.to_csv(OUTPUT_FILE, index=False)
        print(f"\nRFM dataset saved to:" f"\n{OUTPUT_FILE}")
    return data


def main():
    print("\nLoading feature dataset...")
    data = run()
    print("\nRFM Segment Distribution:")
    print(data["RFM_Segment"].value_counts())
    print("\nAverage RFM Scores:")
    print(data[["R_Score", "F_Score", "M_Score", "RFM_Total"]].mean())


if __name__ == "__main__":
    main()
