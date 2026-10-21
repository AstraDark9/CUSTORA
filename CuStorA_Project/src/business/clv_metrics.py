# CuStorA - CLV and Business Metrics
# Calculates customer value and business-priority metrics.

import os

import pandas as pd

# 1. PATHS

INPUT_PATH = "data/processed/customers_recommendations.csv"

OUTPUT_PATH = "data/processed/customers_business.csv"

SUMMARY_PATH = "reports/business_metrics_summary.csv"


# 2. VALUE CALCULATION -- 3-Year Risk-Adjusted Customer Value
#
# This used to be called "Estimated CLV" and computed as
#   AOV x purchase-rate x (1 / churn probability)
# which is a *revenue* projection, not a value model: no cost data
# went into it, no time value of money, and "expected lifetime" was
# an open-ended (if capped) horizon rather than a period the churn
# model was ever asked to predict over.
#
# The column name (EstimatedCLV) is kept for backward compatibility
# with templates/app.py, but what it now holds is a proper, if
# still simplified, discounted-profit model:
#
#   value = sum over year t = 1..VALUE_HORIZON_YEARS of:
#       (expected annual profit in year t)
#       x (probability the customer is still active at the start of year t)
#       / (1 + discount rate)^t
#
# where "expected annual profit" = AOV x purchase rate x gross margin,
# and "probability still active" = (1 - annual churn probability)^(t-1),
# i.e. the standard geometric-survival assumption: a customer who
# hasn't churned by year t-1 has an independent chance of churning
# in each subsequent year.
#
# Two assumptions this makes explicit (previously implicit or
# absent) are configurable constants below rather than buried in the
# formula, per the project review's "Fix CLV properly" note:
#
GROSS_MARGIN_ASSUMPTION = 0.30
# CuStorA's schema has no cost-of-goods or cost-to-serve data, so
# this is a stated business assumption, not something derived from
# the dataset. Change this to whatever the actual business's margin
# is before trusting the absolute numbers (the *ranking* of
# customers by value is unaffected by this constant, since it scales
# every customer's value equally -- only the ranking cutoffs and any
# reported totals depend on it).

ANNUAL_DISCOUNT_RATE = 0.10
# A dollar of profit next year is worth less than a dollar today.
# 10% is a common default for a simplified customer-value model, not
# a value estimated from this business's actual cost of capital.

VALUE_HORIZON_YEARS = 3
# The churn model's PredictedChurnProbability is used here as an
# *annual* churn rate. That's the same assumption the previous
# CLV implementation made implicitly; making the horizon explicit
# and finite (instead of "1 / churn probability years, capped at
# 10") is what turns this into "3-Year Risk-Adjusted Customer
# Value" rather than a claim about a customer's entire remaining
# lifetime, which nothing in CuStorA actually has evidence for.


def calculate_clv(row):
    """
    Estimate 3-Year Risk-Adjusted Customer Value:

        value = sum_{t=1}^{VALUE_HORIZON_YEARS}
                  (AOV x purchase rate x gross margin)
                  x (1 - churn probability)^(t-1)
                  / (1 + discount rate)^t

    "AOV x purchase rate" is this customer's expected *annual
    revenue* going forward (Frequency, their all-time purchase
    count, converted into a rate via TenureYears -- see below for
    why that conversion matters). Multiplying by
    GROSS_MARGIN_ASSUMPTION converts that into expected annual
    *profit*, since value to the business is profit, not revenue.

    The (1 - churn probability)^(t-1) term is the probability this
    customer is still active at the start of year t: certain (1.0)
    for year 1, then shrinking each year by the churn model's
    predicted probability, compounding. Each year's expected profit
    is then discounted back to present value by
    1 / (1 + ANNUAL_DISCOUNT_RATE)^t.

    Why "purchase rate" and not raw Frequency: Frequency is this
    customer's *all-time* purchase count. Multiplying that directly
    by more years would double count -- a customer with 20 purchases
    over a 4-year history isn't projected to make another 20
    purchases next year; dividing by TenureYears first converts
    history into a rate (purchases/year), which is what actually
    projects forward correctly.

    Caveats worth stating explicitly (still a simplified,
    college-project-appropriate estimate, not a full CLV model):
    - GROSS_MARGIN_ASSUMPTION and ANNUAL_DISCOUNT_RATE are stated
      business assumptions (see module-level comments), not values
      derived from data CuStorA has.
    - The churn model's predicted probability is treated as an
      *annual* churn rate over a 3-year horizon, which is an
      assumption CuStorA is making, not something the model was
      actually trained to predict on that fixed a time horizon.
    - This is why the metric is named "3-Year Risk-Adjusted Customer
      Value" everywhere it's displayed, rather than "Customer
      Lifetime Value" -- it doesn't claim to estimate a customer's
      entire remaining relationship with the business, just the
      next three years of it.
    """
    aov = max(float(row["AverageOrderValue"]), 0)
    total_purchases = max(float(row["Frequency"]), 0)
    tenure_years = max(float(row["TenureYears"]), 0.1)
    churn_probability = float(row["PredictedChurnProbability"])
    churn_probability = min(max(churn_probability, 0.0), 1.0)

    # All-time purchase count -> a rate (purchases per year), so we
    # don't project the customer's own purchase history back over the
    # years that produced it.
    purchase_rate_per_year = total_purchases / tenure_years
    expected_annual_profit = aov * purchase_rate_per_year * GROSS_MARGIN_ASSUMPTION
    value = 0.0
    survival_probability = 1.0  # certain to be active at the start of year 1
    for year in range(1, VALUE_HORIZON_YEARS + 1):
        discount_factor = 1 / ((1 + ANNUAL_DISCOUNT_RATE) ** year)
        value += expected_annual_profit * survival_probability * discount_factor
        survival_probability *= 1 - churn_probability
    return max(value, 0)


# 2b. REVENUE AT STAKE -- what is actually lost if this customer leaves
#
# EstimatedCLV above is deliberately discounted by the churn probability, so a
# customer who is very likely to leave can never rank as "high value" on it --
# which made the "Critical" tier (high value AND high risk) impossible to reach.
# For deciding who to save, the relevant quantity is the opposite one: how much
# does this customer normally spend in the prediction window IF they stay?
#
#   RevenueAtStake = their spend per `window_days`, from history
#   ExpectedLoss   = churn probability x RevenueAtStake
#
# Spend is spread over at least one window of tenure, so a customer who joined
# last week is not extrapolated into an enormous run-rate.
DEFAULT_WINDOW_DAYS = 90


def revenue_at_stake(row, window_days=DEFAULT_WINDOW_DAYS):
    window_days = max(float(window_days), 1.0)
    monetary = row.get("Monetary")
    tenure_days = row.get("TenureDays")
    try:
        monetary = float(monetary)
        tenure_days = float(tenure_days)
    except (TypeError, ValueError):
        monetary = tenure_days = float("nan")
    if monetary != monetary or tenure_days != tenure_days:  # NaN: fall back to AOV x rate
        years = max(float(row.get("TenureYears", 0.1)), 0.1)
        rate = max(float(row.get("Frequency", 0)), 0) / years
        return max(float(row.get("AverageOrderValue", 0)), 0) * rate * window_days / 365.0
    return max(monetary, 0.0) * window_days / max(tenure_days, window_days)


# 3. CUSTOMER VALUE CATEGORY


def classify_customer_value(clv, low_threshold, high_threshold):
    if clv >= high_threshold:
        return "High Value"
    elif clv >= low_threshold:
        return "Medium Value"
    else:
        return "Low Value"


# 4. BUSINESS PRIORITY


def determine_business_priority(row):
    clv_category = row["CustomerValue"]
    churn_risk = row["ChurnRisk"]
    customer_state = row["CustomerState"]

    # Highest priority:
    # valuable customers who are likely to churn.

    if clv_category == "High Value" and churn_risk == "High":
        return "Critical Business Priority"
    if clv_category == "High Value" and churn_risk == "Medium":
        return "High Business Priority"
    if clv_category == "Medium Value" and churn_risk == "High":
        return "High Business Priority"
    if customer_state in ["Critical Retention", "Critical Health"]:
        return "High Business Priority"
    if churn_risk == "High":
        return "Medium Business Priority"
    if clv_category == "High Value":
        return "Medium Business Priority"
    return "Normal Business Priority"


# 5. MAIN FUNCTION


def run(data=None, save=True, window_days=DEFAULT_WINDOW_DAYS):
    """
    Calculate CLV/business metrics and (by default) still write
    customers_business.csv (the final dataset the dashboard/customer
    profile pages read) + the business metrics summary + priority
    action list, same as before. `data=None` reads the recommendations
    dataset from disk for standalone/CLI use; pipeline.py instead
    passes the recommendations DataFrame directly, in memory.
    """
    if data is None:
        data = pd.read_csv(INPUT_PATH)
    df = data.copy()
    print(f"Dataset shape: " f"{df.shape[0]} rows x {df.shape[1]} columns")

    # REQUIRED COLUMNS

    required_columns = [
        "CustomerID",
        "AverageOrderValue",
        "Frequency",
        "TenureYears",
        "PredictedChurnProbability",
        "ChurnRisk",
        "CustomerState",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # CLEAN NUMERIC COLUMNS

    numeric_columns = [
        "AverageOrderValue",
        "Frequency",
        "TenureYears",
        "PredictedChurnProbability",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(df[column].median())

    # CALCULATE CLV

    print("\nCalculating 3-Year Risk-Adjusted Customer Value...")
    print(
        f"Assumptions: gross margin={GROSS_MARGIN_ASSUMPTION:.0%}, "
        f"annual discount rate={ANNUAL_DISCOUNT_RATE:.0%}, "
        f"horizon={VALUE_HORIZON_YEARS} years "
        f"(see module docstring in src/business/clv_metrics.py to change these)"
    )
    df["EstimatedCLV"] = df.apply(calculate_clv, axis=1)
    df["EstimatedCLV"] = df["EstimatedCLV"].round(2)

    # REVENUE AT STAKE / EXPECTED LOSS (see section 2b)

    df["RevenueAtStake"] = df.apply(lambda r: revenue_at_stake(r, window_days), axis=1).round(2)
    df["ExpectedLoss"] = (df["RevenueAtStake"] * df["PredictedChurnProbability"]).round(2)
    df["ValueWindowDays"] = int(window_days)

    # VALUE THRESHOLDS (terciles of revenue at stake -- not of the churn-discounted CLV)

    low_threshold = df["RevenueAtStake"].quantile(0.33)
    high_threshold = df["RevenueAtStake"].quantile(0.67)

    # CUSTOMER VALUE CATEGORY

    df["CustomerValue"] = df["RevenueAtStake"].apply(
        lambda value: classify_customer_value(value, low_threshold, high_threshold)
    )

    # BUSINESS PRIORITY

    print("\nDetermining business priorities...")
    df["BusinessPriority"] = df.apply(determine_business_priority, axis=1)

    # Money-based escalation: high stake and high risk rarely coincide (customers
    # who spend a lot also tend to stay), so the rule above alone leaves "Critical"
    # nearly empty. Anyone in the top 1% of expected loss is critical regardless.
    if len(df) >= 50:
        cutoff = df["ExpectedLoss"].quantile(0.99)
        if cutoff > 0:
            df.loc[df["ExpectedLoss"] >= cutoff, "BusinessPriority"] = "Critical Business Priority"

    # VALUE RANK

    df["CLVRank"] = df["EstimatedCLV"].rank(ascending=False, method="min").astype(int)

    # SAVE COMPLETE DATASET

    if save:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)

    # SUMMARY

    print("\nCustomer Value Summary:")
    print(df["CustomerValue"].value_counts())
    print("\nBusiness Priority Summary:")
    print(df["BusinessPriority"].value_counts())

    # BUSINESS METRICS

    total_revenue = df["TotalSpent"].sum()
    average_clv = df["EstimatedCLV"].mean()
    total_estimated_clv = df["EstimatedCLV"].sum()
    average_order_value = df["AverageOrderValue"].mean()
    print("\n" + "=" * 60)
    print("BUSINESS METRICS")
    print("=" * 60)
    print(f"\nTotal Historical Revenue: " f"{total_revenue:,.2f}")
    print(f"Average Order Value: " f"{average_order_value:,.2f}")
    print(f"Average Estimated CLV: " f"{average_clv:,.2f}")
    print(f"Total Estimated CLV: " f"{total_estimated_clv:,.2f}")

    if save:
        # SUMMARY DATASET
        summary_data = pd.DataFrame(
            {
                "Metric": [
                    "Total Customers",
                    "Total Historical Revenue",
                    "Average Order Value",
                    "Value Model: Gross Margin Assumption",
                    "Value Model: Annual Discount Rate",
                    "Value Model: Horizon (Years)",
                    "Average 3-Year Risk-Adjusted Value",
                    "Total 3-Year Risk-Adjusted Value",
                    "High Value Customers",
                    "Medium Value Customers",
                    "Low Value Customers",
                    "Critical Business Priority",
                    "High Business Priority",
                    "Medium Business Priority",
                    "Normal Business Priority",
                ],
                "Value": [
                    len(df),
                    round(total_revenue, 2),
                    round(average_order_value, 2),
                    f"{GROSS_MARGIN_ASSUMPTION:.0%}",
                    f"{ANNUAL_DISCOUNT_RATE:.0%}",
                    VALUE_HORIZON_YEARS,
                    round(average_clv, 2),
                    round(total_estimated_clv, 2),
                    int((df["CustomerValue"] == "High Value").sum()),
                    int((df["CustomerValue"] == "Medium Value").sum()),
                    int((df["CustomerValue"] == "Low Value").sum()),
                    int((df["BusinessPriority"] == "Critical Business Priority").sum()),
                    int((df["BusinessPriority"] == "High Business Priority").sum()),
                    int((df["BusinessPriority"] == "Medium Business Priority").sum()),
                    int((df["BusinessPriority"] == "Normal Business Priority").sum()),
                ],
            }
        )
        os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
        summary_data.to_csv(SUMMARY_PATH, index=False)

    # TOP CUSTOMERS

    print("\nTop 10 Customers by Estimated CLV:")
    top_customers = df.sort_values("EstimatedCLV", ascending=False)
    print(
        top_customers[
            [
                "CustomerID",
                "EstimatedCLV",
                "CustomerValue",
                "ChurnRisk",
                "CustomerState",
                "BusinessPriority",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # ACTION LIST: the customers to act on today

    if save:
        print("\nBuilding priority action list...")
        priority_order = {
            "Critical Business Priority": 0,
            "High Business Priority": 1,
            "Medium Business Priority": 2,
            "Normal Business Priority": 3,
        }
        df["_priority_rank"] = df["BusinessPriority"].map(priority_order).fillna(4)
        action_columns = [
            "CustomerID",
            "BusinessPriority",
            "EstimatedCLV",
            "CustomerValue",
            "ChurnRisk",
            "PredictedChurnProbability",
            "CustomerState",
            "RelationshipHealth" if "RelationshipHealth" in df.columns else "CustomerState",
            "Recommendation" if "Recommendation" in df.columns else "CustomerState",
            "RecommendationAction" if "RecommendationAction" in df.columns else "CustomerState",
            "RecommendationEvidence" if "RecommendationEvidence" in df.columns else "CustomerState",
        ]
        action_columns = list(dict.fromkeys(action_columns))  # de-dupe, keep order
        action_list = df.sort_values(
            by=["_priority_rank", "EstimatedCLV"], ascending=[True, False]
        )[action_columns]
        action_list_path = "reports/priority_action_list.csv"
        action_list.to_csv(action_list_path, index=False)
        print(f"Top 20 customers to act on today (of {len(action_list)} total):")
        print(action_list.head(20).to_string(index=False))
        print("\nFull priority action list saved to:")
        print(action_list_path)
        df.drop(columns=["_priority_rank"], inplace=True)

        print("\nOutput saved to:")
        print(OUTPUT_PATH)
        print("\nBusiness summary saved to:")
        print(SUMMARY_PATH)
    return df


def main():
    print("=" * 60)
    print("CuStorA - CLV and Business Metrics")
    print("=" * 60)
    print("\nLoading recommendation data...")
    run()
    print("\n" + "=" * 60)
    print("CLV AND BUSINESS METRICS COMPLETE")
    print("=" * 60)


# 6. RUN PROGRAM

if __name__ == "__main__":
    main()
