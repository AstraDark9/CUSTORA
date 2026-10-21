# CuStorA - Fuzzy Relationship Health
# Calculates a customer relationship health score
# using Recency, Engagement, and Service Friction.

import os
import numpy as np
import pandas as pd

# 1. PATHS

INPUT_PATH = "data/processed/customers_segmented.csv"
OUTPUT_PATH = "data/processed/customers_health.csv"


# 2. SCORING CONFIGURATION
# These are decision parameters, not statistically fitted values --
# CuStorA does not have enough real transactional history to calibrate
# them against actual churn/retention outcomes. They encode a simple,
# stated business judgement:
#
#   Recency matters most (0.40): a customer who hasn't purchased
#   recently is the clearest single sign that a relationship is
#   cooling off.
#
#   Engagement matters almost as much (0.35): low engagement usually
#   precedes a drop in recency, so it acts as an earlier warning.
#
#   Service friction matters least of the three (0.25): complaints and
#   returns are a real signal, but they're sparser per customer than
#   recency/engagement, so a single incident shouldn't dominate the
#   score the way a person becoming totally inactive should.
#
# If CuStorA were deployed against a real dataset with known retention
# outcomes, these weights should be re-fit (e.g. via logistic
# regression against "did this customer stay or leave") instead of
# left as fixed judgement calls.

RECENCY_WEIGHT = 0.40
ENGAGEMENT_WEIGHT = 0.35
FRICTION_WEIGHT = 0.25

# How much the final score trusts the smooth numeric health calculation
# vs. the fuzzy "good/bad" membership scores. Weighted mostly toward the
# numeric side (0.70) because it uses the full range of each metric,
# with the fuzzy side (0.30) acting as a softer sanity check that
# rewards customers sitting clearly in "good" territory on all three
# metrics at once, rather than just averaging out.
NUMERIC_WEIGHT = 0.70
FUZZY_WEIGHT = 0.30

# Fuzzy membership breakpoints. Recency is in days, engagement is on
# CuStorA's 0-100 EngagementScore, friction is a count of
# complaints + returns. These are reasonable assumptions for the demo
# dataset's purchase cadence, not values derived from a real customer
# base -- for a specific business, they should be set from that
# business's actual typical repurchase cycle and complaint rate.
RECENCY_GOOD_DAYS = 30  # at or below this, recency is fully "good"
RECENCY_BAD_DAYS = 180  # at or above this, recency is fully "bad"
ENGAGEMENT_GOOD_SCORE = 70  # at or above this, engagement is fully "good"
ENGAGEMENT_BAD_SCORE = 30  # at or below this, engagement is fully "bad"
FRICTION_GOOD_COUNT = 1  # at or below this, friction is fully "good"
FRICTION_BAD_COUNT = 5  # at or above this, friction is fully "bad"

# Score thresholds for the 3 health categories shown on the dashboard.
HEALTHY_THRESHOLD = 70
AT_RISK_THRESHOLD = 40


# 3. FUZZY MEMBERSHIP FUNCTIONS


def low_membership(value, low, high):
    """
    Returns how strongly a value belongs to the LOW category.
    Result is between 0 and 1.
    """
    if value <= low:
        return 1.0
    if value >= high:
        return 0.0
    return (high - value) / (high - low)


def high_membership(value, low, high):
    """
    Returns how strongly a value belongs to the HIGH category.
    Result is between 0 and 1.
    """
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    return (value - low) / (high - low)


# 4. NORMALIZATION


def normalize(value, minimum, maximum):
    """
    Converts a value into a 0-1 range.
    """
    if maximum == minimum:
        return 0.0
    return np.clip((value - minimum) / (maximum - minimum), 0, 1)


# 5. CALCULATE RELATIONSHIP HEALTH


def calculate_health(row, max_recency, max_friction, friction_observed=True):

    # RECENCY
    # Lower recency = customer purchased recently
    # Higher recency = customer has been inactive

    recency = row["Recency"]
    recency_health = 1 - normalize(recency, 0, max_recency)

    # ENGAGEMENT

    engagement = row["EngagementScore"]
    engagement_health = normalize(engagement, 0, 100)

    # SERVICE FRICTION
    # Complaints + Returns represent friction.

    complaints = row["Complaints"]
    returns = row["Returns"]
    friction = complaints + returns
    friction_health = 1 - normalize(friction, 0, max_friction)

    # FUZZY MEMBERSHIP VALUES
    # See the SCORING CONFIGURATION section above for why these
    # particular breakpoints were chosen.

    recency_good = low_membership(recency, RECENCY_GOOD_DAYS, RECENCY_BAD_DAYS)
    engagement_good = high_membership(engagement, ENGAGEMENT_BAD_SCORE, ENGAGEMENT_GOOD_SCORE)
    friction_good = low_membership(friction, FRICTION_GOOD_COUNT, FRICTION_BAD_COUNT)

    # FUZZY HEALTH SCORE

    # If the data has no complaint/return information at all (every
    # customer has zero friction, e.g. a plain transaction ledger), friction
    # says nothing -- leaving it in would hand every customer a free 25
    # points. Drop it and re-weight recency/engagement to sum to 1 instead.
    if friction_observed:
        w_recency, w_engagement, w_friction = RECENCY_WEIGHT, ENGAGEMENT_WEIGHT, FRICTION_WEIGHT
    else:
        total = RECENCY_WEIGHT + ENGAGEMENT_WEIGHT
        w_recency, w_engagement, w_friction = RECENCY_WEIGHT / total, ENGAGEMENT_WEIGHT / total, 0.0
    health_score = (
        w_recency * recency_health
        + w_engagement * engagement_health
        + w_friction * friction_health
    )
    fuzzy_support = (
        w_recency * recency_good
        + w_engagement * engagement_good
        + w_friction * friction_good
    )

    # Combine numerical health with fuzzy membership.
    final_score = NUMERIC_WEIGHT * health_score + FUZZY_WEIGHT * fuzzy_support
    final_score = np.clip(final_score * 100, 0, 100)
    return final_score


# 6. HEALTH CATEGORY


def classify_health(score):
    if score >= HEALTHY_THRESHOLD:
        return "Healthy"
    elif score >= AT_RISK_THRESHOLD:
        return "At Risk"
    else:
        return "Critical"


def compute_relationship_health(df):
    """
    Pure transform: adds RelationshipHealthScore/RelationshipHealth to
    a copy of the segmented customer dataframe. Split out from main()
    so pipeline.py can call it directly on an in-memory DataFrame
    without a disk round-trip.
    """
    df = df.copy()

    # CHECK REQUIRED COLUMNS

    required_columns = ["Recency", "EngagementScore", "Complaints", "Returns"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # PREPARE VALUES

    df["Recency"] = pd.to_numeric(df["Recency"], errors="coerce")
    df["EngagementScore"] = pd.to_numeric(df["EngagementScore"], errors="coerce")
    df["Complaints"] = pd.to_numeric(df["Complaints"], errors="coerce")
    df["Returns"] = pd.to_numeric(df["Returns"], errors="coerce")

    # HANDLE MISSING VALUES

    df["Recency"] = df["Recency"].fillna(df["Recency"].median())
    df["EngagementScore"] = df["EngagementScore"].fillna(df["EngagementScore"].median())
    df["Complaints"] = df["Complaints"].fillna(0)
    df["Returns"] = df["Returns"].fillna(0)

    # MAXIMUM VALUES

    max_recency = max(df["Recency"].max(), 1)
    friction_values = df["Complaints"] + df["Returns"]
    max_friction = max(friction_values.max(), 1)
    friction_observed = bool(friction_values.nunique() > 1)

    # CALCULATE HEALTH

    print("\nCalculating relationship health...")
    df["RelationshipHealthScore"] = df.apply(
        lambda row: calculate_health(row, max_recency, max_friction, friction_observed),
        axis=1
    )
    df["RelationshipHealth"] = df["RelationshipHealthScore"].apply(classify_health)
    return df


def run(data=None, save=True):
    """
    Score relationship health and (by default) still write
    customers_health.csv, same as before. `data=None` reads the
    segmented dataset from disk for standalone/CLI use; pipeline.py
    instead passes the segmented DataFrame directly, in memory.
    """
    if data is None:
        data = pd.read_csv(INPUT_PATH)
    print(f"Dataset shape: " f"{data.shape[0]} rows x {data.shape[1]} columns")
    df = compute_relationship_health(data)

    # SAVE RESULT

    if save:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)

    # SUMMARY

    print("\nRelationship Health Summary:")
    print(df["RelationshipHealth"].value_counts())
    print("\nAverage Health Score:")
    print(round(df["RelationshipHealthScore"].mean(), 2))
    print("\nSample results:")
    print(
        df[
            [
                "CustomerID",
                "Recency",
                "EngagementScore",
                "Complaints",
                "Returns",
                "RelationshipHealthScore",
                "RelationshipHealth",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
    if save:
        print("\nOutput saved to:")
        print(OUTPUT_PATH)
    return df


# 7. MAIN FUNCTION


def main():
    print("=" * 60)
    print("CuStorA - Fuzzy Relationship Health")
    print("=" * 60)
    print("\nLoading customer data...")
    run()
    print("\n" + "=" * 60)
    print("FUZZY RELATIONSHIP HEALTH COMPLETE")
    print("=" * 60)


# 8. RUN PROGRAM

if __name__ == "__main__":
    main()
