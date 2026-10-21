"""
Generates a synthetic customer dataset for CuStorA.

Used when the user doesn't have their own CSV and asks
CuStorA to generate sample data instead.

DESIGN NOTE (see project review notes, issues #2, #8, #14):

The previous version generated every column independently (random
TotalPurchases, random TotalSpent, random EngagementScore, ...) and
then derived Churn directly from a visible rule over those same raw
columns:

    churn_probability = 0.20 + (engagement < 35) * 0.25
                              + (complaints >= 3) * 0.15 + ...

Two problems followed from that:

1. Independent columns meant the dataset didn't look like real
   customer behavior -- in reality, frequent buyers tend to spend
   more in total, engaged customers tend to complain/return less,
   longer-tenured customers have had more time to accumulate
   purchases, and so on. K-Means and the churn model were learning
   from a fairly artificial customer universe.

2. Deriving churn from a literal threshold rule over the exact same
   fields fed to the model turns "predict churn" into "rediscover the
   rule that generated this column," which is a much easier and less
   meaningful problem than real churn prediction.

This version instead gives each customer a hidden "relationship
quality" trait (not saved to the output CSV) and uses it to pull
several observed columns in a consistent direction, with independent
noise layered on top of each one. Purchases, spend, engagement,
complaints, and returns are correlated with each other and with
tenure the way a real customer base would be, and Churn is generated
from a noisy combination of that hidden trait plus the observed
fields -- related to them, but not perfectly reconstructable from
them, since real-world churn always has causes a model never
observes.

Age is deliberately generated independently of the hidden trait and
of Churn, matching how age plausibly is *not* a strong churn driver
in reality. Column-provenance is intentionally still visible here
(see the comment above churn_probability below) so anyone using this
generator for a viva/report can see exactly what is and isn't a real
signal in the data it produces.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_OUTPUT_PATH = Path("data/demo/customers_demo.csv")


def generate_customer_data(n_customers=1000, seed=42):
    """Build a random but internally-correlated customer dataset."""
    rng = np.random.default_rng(seed)
    customer_ids = [f"CUST{1001 + i}" for i in range(n_customers)]

    # Age is intentionally independent of everything else, including
    # Churn -- see module docstring.
    ages = rng.integers(18, 65, n_customers)
    genders = rng.choice(["Male", "Female"], size=n_customers)
    registration_dates = pd.date_range(start="2021-01-01", end="2025-12-31", periods=n_customers)
    tenure_years = (pd.Timestamp("2026-07-31") - pd.Series(registration_dates)).dt.days / 365.25

    # HIDDEN "RELATIONSHIP QUALITY" TRAIT
    # Not saved to the CSV. A single underlying trait per customer,
    # in [0, 1], that several observed columns are correlated with --
    # this is what makes the observed columns correlated with *each
    # other* too, the way real customer behavior is, instead of each
    # being drawn independently.
    relationship_quality = rng.beta(2, 2, n_customers)

    # PURCHASE BEHAVIOR
    # Higher-quality, longer-tenured customers accumulate more
    # purchases; each customer also gets independent noise so the
    # relationship isn't a deterministic formula.
    expected_purchases = 2 + relationship_quality * 25 + tenure_years.to_numpy() * 3
    total_purchases = rng.poisson(np.clip(expected_purchases, 1, None)).clip(min=1)

    # Spend per purchase also rises with relationship quality (better
    # customers trend toward higher-value orders, not just more of
    # them), with its own independent noise.
    order_value_base = 300 + relationship_quality * 3000
    order_value_noise = rng.lognormal(mean=0, sigma=0.5, size=n_customers)
    average_order_value = np.round(order_value_base * order_value_noise, 2)
    total_spent = np.round(average_order_value * total_purchases, 2)

    # ENGAGEMENT, COMPLAINTS, RETURNS
    # Engagement rises with relationship quality; complaints/returns
    # fall with it -- each still with substantial independent noise
    # so quality alone doesn't fully determine any single column.
    engagement_score = np.clip(
        10 + relationship_quality * 80 + rng.normal(0, 12, n_customers), 0, 100
    ).round(2)
    complaint_rate = np.clip(2.2 - relationship_quality * 1.8, 0.05, None)
    return_rate = np.clip(1.8 - relationship_quality * 1.4, 0.05, None)
    complaints = rng.poisson(complaint_rate)
    returns = rng.poisson(return_rate)
    discount_usage = np.round(
        np.clip(rng.normal(50 - relationship_quality * 15, 25, n_customers), 0, 100), 2
    )
    last_purchase_dates = rng.choice(pd.date_range("2025-01-01", "2026-07-31"), size=n_customers)

    # CHURN
    # Related to relationship_quality (a hidden trait, not a column
    # the model ever sees) and to a handful of observed signals, but
    # with a wide noise band -- so the model has real, non-circular
    # signal to find, without the label being a perfect function of
    # its own inputs. The exact weights below are intentionally
    # documented, not hidden, so a report/viva can cite them honestly
    # as "how the demo data was constructed," not "how real churn
    # works."
    churn_logit = (
        -0.85
        - 2.0 * relationship_quality
        + 0.010 * complaints
        + 0.008 * returns
        - 0.004 * total_purchases
        + rng.normal(0, 1.0, n_customers)  # unobserved causes, same as real churn
    )
    churn_probability = 1 / (1 + np.exp(-churn_logit))
    churn = rng.binomial(1, churn_probability)
    return pd.DataFrame(
        {
            "CustomerID": customer_ids,
            "Age": ages,
            "Gender": genders,
            "RegistrationDate": registration_dates,
            "LastPurchaseDate": last_purchase_dates,
            "TotalPurchases": total_purchases,
            "TotalSpent": total_spent,
            "AverageOrderValue": average_order_value,
            "EngagementScore": engagement_score,
            "Complaints": complaints,
            "Returns": returns,
            "DiscountUsage": discount_usage,
            "Churn": churn,
        }
    )


def save_demo_dataset(n_customers=1000, output_path=DEFAULT_OUTPUT_PATH):
    data = generate_customer_data(n_customers)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Generated dataset saved to: {output_path}")
    print(f"Rows: {len(data)}, Columns: {len(data.columns)}")
    print(f"Churn rate: {data['Churn'].mean():.1%}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic customer dataset.")
    parser.add_argument("--count", type=int, default=1000, help="Number of customers to generate")
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Output CSV path"
    )
    args = parser.parse_args()
    save_demo_dataset(args.count, args.output)
