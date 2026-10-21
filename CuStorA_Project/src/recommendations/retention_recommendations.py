# CuStorA - Retention Recommendation Engine
# Generates rule-based retention recommendations
# using customer state and behavior.
#
# Each recommendation now carries the *evidence* that triggered it
# (the specific signals that were checked) and a *confidence* label
# derived from how many of those signals actually fired. This makes
# the engine's output auditable -- "why did it say this?" -- instead
# of a single opaque sentence, without pretending the underlying logic
# is anything more than transparent, inspectable business rules.

import os

import pandas as pd

# 1. PATHS

INPUT_PATH = "data/processed/customers_state.csv"

OUTPUT_PATH = "data/processed/customers_recommendations.csv"


# 2. RECOMMENDATION RULES


def _evidence_strength_from_evidence(evidence):
    """
    Labeled "Evidence Strength," not "Confidence" (see project review
    notes, issue #6). "Confidence" implies a statistical estimate --
    e.g. "the model is 85% confident" -- and this isn't one. It's a
    count of how many independent rule conditions fired for this
    customer: 2+ signals -> "Strong evidence", 1 signal -> "Moderate
    evidence", 0 extra signals (the recommendation came from the
    customer's state alone) -> "State-based". That's a meaningfully
    different, weaker claim than "confidence," and the label should
    say so.
    """
    if len(evidence) >= 2:
        return "Strong evidence"
    if len(evidence) == 1:
        return "Moderate evidence"
    return "State-based"


def generate_recommendation(row):
    state = row["CustomerState"]
    recency = row["Recency"]
    engagement = row["EngagementScore"]
    complaints = row["Complaints"]
    returns = row["Returns"]

    # CRITICAL RETENTION

    if state == "Critical Retention":
        if recency > 180 and engagement < 40:
            evidence = [
                f"{int(recency)} days since last purchase (>180)",
                f"Engagement score {engagement:.0f} (<40)",
            ]
            return (
                "Urgent re-engagement campaign",
                "Contact customer with a personalized "
                "win-back offer and product recommendations.",
                "Urgent",
                evidence,
            )
        elif complaints >= 3:
            evidence = [f"{int(complaints)} complaints on record (>=3)"]
            return (
                "Service recovery",
                "Contact customer to resolve service issues " "and provide a recovery offer.",
                "Urgent",
                evidence,
            )
        elif returns >= 3:
            evidence = [f"{int(returns)} returns on record (>=3)"]
            return (
                "Return issue intervention",
                "Review returned products and offer assistance " "or suitable alternatives.",
                "Urgent",
                evidence,
            )
        else:
            evidence = ["Customer state classified as Critical Retention"]
            return (
                "Priority retention campaign",
                "Contact customer with a personalized " "retention offer.",
                "Urgent",
                evidence,
            )

    # HIGH RISK

    if state == "High Risk":
        if engagement < 40:
            evidence = [f"Engagement score {engagement:.0f} (<40)"]
            return (
                "Re-engagement campaign",
                "Send personalized content or an offer " "to increase customer engagement.",
                "High",
                evidence,
            )
        elif recency > 120:
            evidence = [f"{int(recency)} days since last purchase (>120)"]
            return (
                "Win-back campaign",
                "Send a personalized reminder and " "limited-time return offer.",
                "High",
                evidence,
            )
        else:
            evidence = ["Customer state classified as High Risk"]
            return (
                "Retention offer",
                "Provide a personalized incentive " "to encourage another purchase.",
                "High",
                evidence,
            )

    # HIDDEN RISK

    if state == "Hidden Risk":
        evidence = ["Churn model flags risk not yet visible in relationship health score"]
        return (
            "Early retention intervention",
            "Monitor the customer closely and provide " "a personalized engagement campaign.",
            "High",
            evidence,
        )

    # WATCH

    if state == "Watch":
        if engagement < 40:
            evidence = [f"Engagement score {engagement:.0f} (<40)"]
            return (
                "Increase engagement",
                "Send personalized recommendations, " "content, or a targeted offer.",
                "Medium",
                evidence,
            )
        elif recency > 120:
            evidence = [f"{int(recency)} days since last purchase (>120)"]
            return (
                "Re-engagement reminder",
                "Send a reminder or personalized offer " "to encourage another purchase.",
                "Medium",
                evidence,
            )
        elif complaints >= 2:
            evidence = [f"{int(complaints)} complaints on record (>=2)"]
            return (
                "Service improvement",
                "Follow up on customer complaints " "and improve the service experience.",
                "Medium",
                evidence,
            )
        else:
            evidence = ["Customer state classified as Watch"]
            return (
                "Relationship monitoring",
                "Continue monitoring customer behavior " "and engagement.",
                "Medium",
                evidence,
            )

    # CRITICAL HEALTH

    if state == "Critical Health":
        if complaints >= 3:
            evidence = [f"{int(complaints)} complaints on record (>=3)"]
            return (
                "Service recovery",
                "Resolve outstanding customer issues " "and follow up personally.",
                "High",
                evidence,
            )
        elif recency > 180:
            evidence = [f"{int(recency)} days since last purchase (>180)"]
            return (
                "Win-back campaign",
                "Use a personalized offer to bring " "the inactive customer back.",
                "High",
                evidence,
            )
        else:
            evidence = ["Relationship health score classified as Critical"]
            return (
                "Relationship recovery",
                "Increase engagement and address " "potential customer concerns.",
                "High",
                evidence,
            )

    # MONITOR

    if state == "Monitor":
        evidence = ["No urgent signal -- routine monitoring"]
        return (
            "Monitor customer",
            "Continue monitoring customer behavior " "and churn indicators.",
            "Low",
            evidence,
        )

    # STABLE

    if state == "Stable":
        evidence = ["Customer state classified as Stable"]
        return (
            "Loyalty and engagement",
            "Maintain the relationship with loyalty " "benefits and personalized recommendations.",
            "Low",
            evidence,
        )

    # DEFAULT

    return (
        "Monitor customer",
        "Continue monitoring customer behavior.",
        "Low",
        ["No customer state available"],
    )


def generate_recommendation_row(row):
    """
    Wraps generate_recommendation and adds a derived Evidence Strength
    label plus a display-ready, pipe-joined Evidence string (easy to
    store in a flat CSV and to split back apart in the UI).
    """
    title, action, priority, evidence = generate_recommendation(row)
    evidence_strength = _evidence_strength_from_evidence(evidence)
    return title, action, priority, " | ".join(evidence), evidence_strength, len(evidence)


def run(data=None, save=True):
    """
    Generate retention recommendations and (by default) still write
    customers_recommendations.csv + the recommendation summary report,
    same as before. `data=None` reads the state dataset from disk for
    standalone/CLI use; pipeline.py instead passes the state DataFrame
    directly, in memory.
    """
    if data is None:
        data = pd.read_csv(INPUT_PATH)
    df = data
    print(f"Dataset shape: " f"{df.shape[0]} rows x {df.shape[1]} columns")

    # CHECK REQUIRED COLUMNS

    required_columns = [
        "CustomerID",
        "CustomerState",
        "ChurnRisk",
        "Recency",
        "EngagementScore",
        "Complaints",
        "Returns",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # GENERATE RECOMMENDATIONS

    print("\nGenerating retention recommendations...")
    df = df.copy()
    recommendations = df.apply(generate_recommendation_row, axis=1)

    # STORE RESULTS

    df[
        [
            "Recommendation",
            "RecommendationAction",
            "RecommendationPriority",
            "RecommendationEvidence",
            "RecommendationEvidenceStrength",
            "RecommendationSignalCount",
        ]
    ] = pd.DataFrame(recommendations.tolist(), index=df.index)

    # SAVE DATA

    if save:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)

    # SUMMARY

    print("\nRecommendation Priority:")
    print(df["RecommendationPriority"].value_counts())
    print("\nRecommendation Evidence Strength:")
    print(df["RecommendationEvidenceStrength"].value_counts())
    print("\nTop Recommendations:")
    print(df["Recommendation"].value_counts().head(10))

    # SAMPLE RESULTS

    print("\nSample Customer Recommendations:")
    sample_columns = [
        "CustomerID",
        "CustomerState",
        "ChurnRisk",
        "Recommendation",
        "RecommendationAction",
        "RecommendationPriority",
        "RecommendationEvidenceStrength",
    ]
    print(df[sample_columns].head(15).to_string(index=False))

    if save:
        # SUMMARY FILE
        recommendation_summary = (
            df.groupby("Recommendation")
            .agg(
                CustomerCount=("Recommendation", "count"),
                StrongEvidenceCount=(
                    "RecommendationEvidenceStrength",
                    lambda s: (s == "Strong evidence").sum(),
                ),
            )
            .reset_index()
            .sort_values(by="CustomerCount", ascending=False)
        )
        summary_path = "reports/retention_recommendation_summary.csv"
        os.makedirs("reports", exist_ok=True)
        recommendation_summary.to_csv(summary_path, index=False)
        print("\nOutput saved to:")
        print(OUTPUT_PATH)
        print("\nRecommendation summary saved to:")
        print(summary_path)
    return df


# 3. MAIN FUNCTION


def main():
    print("=" * 60)
    print("CuStorA - Retention Recommendation Engine")
    print("=" * 60)
    print("\nLoading customer state data...")
    run()
    print("\n" + "=" * 60)
    print("RETENTION RECOMMENDATION ENGINE COMPLETE")
    print("=" * 60)


# 4. RUN PROGRAM

if __name__ == "__main__":
    main()
