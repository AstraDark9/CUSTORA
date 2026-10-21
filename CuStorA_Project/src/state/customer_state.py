# CuStorA - Customer State and Risk
# Combines churn risk, relationship health,
# and customer segmentation into one customer state.

import os
import json

import joblib
import pandas as pd

from src.churn import feature_contract

# 1. PATHS

DATA_PATH = "data/processed/customers_health.csv"

MODEL_PATH = "models/churn/best_churn_model.joblib"

OUTPUT_PATH = "data/processed/customers_state.csv"


# 2. CHURN MODEL FEATURES

COLUMNS_TO_DROP = [
    "CustomerID",
    "RegistrationDate",
    "LastPurchaseDate",
    "Churn",
    "Cluster",
    "SegmentName",
    "RFM_Segment",
    "RelationshipHealthScore",
    "RelationshipHealth",
]


# 3. STATE / RISK CONFIGURATION
# Thresholds are loaded from reports/churn/decision_thresholds.json,
# produced by churn_models.py's select_operating_threshold /
# save_decision_thresholds from this run's actual calibrated
# probability distribution and test-set precision/recall tradeoff --
# not hardcoded round numbers (see project review notes, issue #5,
# and the docstrings in churn_models.py for the reasoning).
#
# The constants below are fallback defaults only, used if that file
# is missing -- e.g. in unit tests that call these functions directly
# without running the full pipeline first.
DECISION_THRESHOLDS_PATH = "reports/churn/decision_thresholds.json"

DEFAULT_HIGH_RISK_THRESHOLD = 0.60
DEFAULT_MEDIUM_RISK_THRESHOLD = 0.40
DEFAULT_OPERATING_THRESHOLD = 0.50


def load_decision_thresholds(path=DECISION_THRESHOLDS_PATH):
    """Load this run's data-driven thresholds, or fall back to fixed
    defaults if churn_models.py hasn't produced them yet."""
    if os.path.exists(path):
        with open(path) as f:
            thresholds = json.load(f)
        print(f"Using data-driven decision thresholds from: {path}")
        return {
            "high_risk_threshold": thresholds.get(
                "high_risk_threshold", DEFAULT_HIGH_RISK_THRESHOLD
            ),
            "medium_risk_threshold": thresholds.get(
                "medium_risk_threshold", DEFAULT_MEDIUM_RISK_THRESHOLD
            ),
            "operating_threshold": thresholds.get(
                "operating_threshold", DEFAULT_OPERATING_THRESHOLD
            ),
        }
    print(
        f"No decision_thresholds.json found at {path} -- using fixed fallback "
        f"defaults (High >= {DEFAULT_HIGH_RISK_THRESHOLD}, "
        f"Medium >= {DEFAULT_MEDIUM_RISK_THRESHOLD}, "
        f"PredictedChurn cutoff = {DEFAULT_OPERATING_THRESHOLD})."
    )
    return {
        "high_risk_threshold": DEFAULT_HIGH_RISK_THRESHOLD,
        "medium_risk_threshold": DEFAULT_MEDIUM_RISK_THRESHOLD,
        "operating_threshold": DEFAULT_OPERATING_THRESHOLD,
    }


# 4. CUSTOMER STATE RULES
#
# high_risk_threshold / medium_risk_threshold are accepted as explicit
# arguments (defaulting to the fixed fallback constants above) rather
# than read off a module-level global. A global mutated by main() at
# the start of each run (as this used to be) is a shared, process-wide
# piece of state -- if two pipeline runs for different uploaded
# datasets were ever in flight in the same process at once (e.g. once
# /run-analysis moves off the request thread), one run's thresholds
# could silently overwrite the other's mid-calculation. Passing
# thresholds as arguments makes each call self-contained instead.


def determine_customer_state(
    health,
    churn_probability,
    high_risk_threshold=DEFAULT_HIGH_RISK_THRESHOLD,
    medium_risk_threshold=DEFAULT_MEDIUM_RISK_THRESHOLD,
):
    """
    Determines the overall customer state using
    relationship health and predicted churn probability.
    """

    # High churn risk
    if churn_probability >= high_risk_threshold:
        if health == "Critical":
            return "Critical Retention"
        elif health == "At Risk":
            return "High Risk"
        else:
            return "Hidden Risk"

    # Medium churn risk
    elif churn_probability >= medium_risk_threshold:
        if health == "Critical":
            return "Critical Health"
        elif health == "At Risk":
            return "Watch"
        else:
            return "Monitor"

    # Low churn risk
    else:
        if health == "Critical":
            return "Critical Health"
        elif health == "At Risk":
            return "Watch"
        else:
            return "Stable"


# 5. CHURN RISK CATEGORY


def classify_churn_risk(
    probability,
    high_risk_threshold=DEFAULT_HIGH_RISK_THRESHOLD,
    medium_risk_threshold=DEFAULT_MEDIUM_RISK_THRESHOLD,
):
    if probability >= high_risk_threshold:
        return "High"
    elif probability >= medium_risk_threshold:
        return "Medium"
    else:
        return "Low"


def compute_customer_state(df, model, high_risk_threshold, medium_risk_threshold, operating_threshold):
    """
    Pure transform: adds PredictedChurnProbability/PredictedChurn/
    ChurnRisk/CustomerState to a copy of the health-scored dataframe,
    using an already-loaded churn model. Split out from main() so
    pipeline.py can call it directly on an in-memory DataFrame and an
    already-loaded model, without a disk round-trip for either.
    """
    df = df.copy()

    # CHECK REQUIRED COLUMNS

    required_columns = [
        "CustomerID",
        "RelationshipHealthScore",
        "RelationshipHealth",
        "Cluster",
        "SegmentName",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # PREPARE FEATURES

    print("\nPreparing customer features...")
    X = feature_contract.select_model_features(df, COLUMNS_TO_DROP, model)

    # PREDICT CHURN PROBABILITY

    print("\nCalculating churn probability...")
    churn_probability = model.predict_proba(X)[:, 1]
    df["PredictedChurnProbability"] = churn_probability

    # CHURN PREDICTION

    df["PredictedChurn"] = (df["PredictedChurnProbability"] >= operating_threshold).astype(int)

    # CHURN RISK CATEGORY

    df["ChurnRisk"] = df["PredictedChurnProbability"].apply(
        lambda p: classify_churn_risk(p, high_risk_threshold, medium_risk_threshold)
    )

    # CUSTOMER STATE

    print("\nDetermining customer states...")
    df["CustomerState"] = df.apply(
        lambda row: determine_customer_state(
            row["RelationshipHealth"],
            row["PredictedChurnProbability"],
            high_risk_threshold,
            medium_risk_threshold,
        ),
        axis=1,
    )

    # ROUND PROBABILITY

    df["PredictedChurnProbability"] = df["PredictedChurnProbability"].round(4)
    return df


def run(data=None, model=None, save=True):
    """
    Predict churn and determine customer state, and (by default) still
    write customers_state.csv + the state summary report, same as
    before. `data=None` reads the health-scored dataset from disk, and
    `model=None` loads the churn model from disk -- both for
    standalone/CLI use. pipeline.py instead passes the health-scored
    DataFrame and the already-loaded (trained-or-reused) churn model
    directly, avoiding a disk round-trip and a second model load.
    """
    if data is None:
        data = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: " f"{data.shape[0]} rows x {data.shape[1]} columns")

    print("\nLoading decision thresholds...")
    thresholds = load_decision_thresholds()
    high_risk_threshold = thresholds["high_risk_threshold"]
    medium_risk_threshold = thresholds["medium_risk_threshold"]
    operating_threshold = thresholds["operating_threshold"]

    if model is None:
        print("\nLoading churn model...")
        model = joblib.load(MODEL_PATH)
        print("Churn model loaded successfully.")

    df = compute_customer_state(
        data, model, high_risk_threshold, medium_risk_threshold, operating_threshold
    )

    if save:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)

    # DISPLAY SUMMARY

    print("\n" + "=" * 60)
    print("CUSTOMER STATE SUMMARY")
    print("=" * 60)
    print("\nChurn Risk:")
    print(df["ChurnRisk"].value_counts())
    print("\nCustomer State:")
    print(df["CustomerState"].value_counts())

    # DISPLAY SAMPLE

    print("\nSample Customer Results:")
    sample_columns = [
        "CustomerID",
        "SegmentName",
        "RelationshipHealthScore",
        "RelationshipHealth",
        "PredictedChurnProbability",
        "ChurnRisk",
        "CustomerState",
    ]
    print(df[sample_columns].head(15).to_string(index=False))

    if save:
        # SAVE STATE SUMMARY
        state_summary = df["CustomerState"].value_counts().reset_index()
        state_summary.columns = ["CustomerState", "CustomerCount"]
        summary_path = "reports/customer_state_summary.csv"
        os.makedirs("reports", exist_ok=True)
        state_summary.to_csv(summary_path, index=False)
        print("\nOutput saved to:")
        print(OUTPUT_PATH)
        print("\nState summary saved to:")
        print(summary_path)
    return df


# 6. MAIN FUNCTION


def main():
    print("=" * 60)
    print("CuStorA - Customer State and Risk")
    print("=" * 60)
    print("\nLoading customer health data...")
    run()
    print("\n" + "=" * 60)
    print("CUSTOMER STATE ANALYSIS COMPLETE")
    print("=" * 60)


# 7. RUN PROGRAM

if __name__ == "__main__":
    main()
