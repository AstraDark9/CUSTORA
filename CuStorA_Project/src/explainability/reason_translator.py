# CuStorA - Plain-English Churn Reasons
#
# The SHAP explainer produces rows like:
#   Feature="num__EngagementScore", SHAP_Value=-0.13
# which is accurate but meaningless to a non-technical reader.
#
# This module turns rows like that into something a retention rep
# could actually read and act on, e.g.:
#   "Engagement has dropped -- this customer rarely interacts anymore."
#   Suggested action: "Send a re-engagement email with tailored
#   product recommendations."
#
# It also collapses near-duplicate signals (e.g. Complaints and
# ComplaintRate both point at "service friction") down to one reason,
# so the customer view doesn't repeat the same idea twice.

# 1. FEATURE -> PLAIN-ENGLISH CONCEPT
#
# Maps a raw model feature name to:
#   - concept:     a short internal key used to dedupe similar features
#   - label:       what a human would call this factor
#   - risk_text:   sentence used when this factor is PUSHING churn risk UP
#   - safe_text:   sentence used when this factor is PULLING churn risk DOWN
#   - action:      suggested next step, only shown for risk-increasing reasons
#
# {value} is filled in from the customer's actual data where available.

# Currency symbol shown in reason texts; app.py sets it from the current run.
CURRENCY_SYMBOL = "\u20b9"

FEATURE_CONCEPTS = {
    "Recency": {
        "concept": "recency",
        "label": "Time since last purchase",
        "risk_text": "Hasn't purchased in {Recency:.0f} days -- longer than most customers.",
        "safe_text": "Purchased recently, only {Recency:.0f} days ago.",
        "action": "Send a personalized win-back offer before the gap grows further.",
    },
    "Frequency": {
        "concept": "frequency",
        "label": "Purchase frequency",
        "risk_text": "Only {Frequency:.0f} purchases on record -- well below a typical customer.",
        "safe_text": "Buys often ({Frequency:.0f} purchases on record).",
        "action": "Offer a small incentive (bundle deal or loyalty points) to prompt another order.",
    },
    "Monetary": {
        "concept": "monetary",
        "label": "Total spend",
        "risk_text": "Total spend of ₹{Monetary:,.0f} is on the lower end.",
        "safe_text": "Has spent ₹{Monetary:,.0f} to date -- a strong contributor.",
        "action": "Highlight higher-value products or a loyalty tier to grow spend.",
    },
    "EngagementScore": {
        "concept": "engagement",
        "label": "Engagement",
        "risk_text": "Engagement has dropped -- this customer rarely interacts anymore (score {EngagementScore:.0f}/100).",
        "safe_text": "Stays actively engaged (score {EngagementScore:.0f}/100).",
        "action": "Send a re-engagement email with tailored product recommendations.",
    },
    "Complaints": {
        "concept": "friction",
        "label": "Complaints",
        "risk_text": "Has filed {Complaints:.0f} complaint(s) -- a sign of a service problem.",
        "safe_text": "Has had few or no complaints.",
        "action": "Reach out proactively to resolve any open service issues.",
    },
    "ComplaintRate": {
        "concept": "friction",
        "label": "Complaint rate",
        "risk_text": "Complains relatively often compared to how much they buy.",
        "safe_text": "Complains rarely relative to their purchase history.",
        "action": "Reach out proactively to resolve any open service issues.",
    },
    "Returns": {
        "concept": "returns",
        "label": "Returns",
        "risk_text": "Has returned {Returns:.0f} order(s) -- may be unhappy with product fit or quality.",
        "safe_text": "Rarely returns items.",
        "action": "Check in on product satisfaction and offer sizing/fit guidance if relevant.",
    },
    "ReturnRate": {
        "concept": "returns",
        "label": "Return rate",
        "risk_text": "Returns a relatively high share of what they order.",
        "safe_text": "Keeps most of what they order.",
        "action": "Check in on product satisfaction and offer sizing/fit guidance if relevant.",
    },
    "DiscountUsage": {
        "concept": "discount_usage",
        "label": "Discount reliance",
        "risk_text": "Relies heavily on discounts ({DiscountUsage:.0f}% of activity) -- price-sensitive and flight-risk without one.",
        "safe_text": "Doesn't rely much on discounts to keep buying.",
        "action": "Offer a targeted, time-limited discount rather than a broad markdown.",
    },
    "TenureYears": {
        "concept": "tenure",
        "label": "Tenure",
        "risk_text": "Still a fairly new customer ({TenureYears:.1f} years) -- hasn't built loyalty yet.",
        "safe_text": "A long-tenured customer ({TenureYears:.1f} years).",
        "action": "Welcome-style engagement: introduce loyalty perks early to build the relationship.",
    },
    "TenureDays": {
        "concept": "tenure",
        "label": "Tenure",
        "risk_text": "A fairly new customer ({TenureDays:.0f} days) -- hasn't built loyalty yet.",
        "safe_text": "A long-standing customer ({TenureDays:.0f} days).",
        "action": "Welcome-style engagement: introduce loyalty perks early to build the relationship.",
    },
    "AverageOrderValue": {
        "concept": "order_value",
        "label": "Average order value",
        "risk_text": "Average order value of ₹{AverageOrderValue:,.0f} is on the lower end.",
        "safe_text": "Healthy average order value (₹{AverageOrderValue:,.0f}).",
        "action": "Suggest bundles or complementary products to lift basket size.",
    },
    "RecencyOverGap": {
        "concept": "overdue",
        "label": "Overdue vs. their usual rhythm",
        "risk_text": "Is {RecencyOverGap:.1f}x overdue compared with how often they normally buy.",
        "safe_text": "Still buying within their usual rhythm.",
        "action": "Send a timely reminder or personalised offer -- they are past their usual reorder point.",
    },
    "AvgGapDays": {
        "concept": "rhythm",
        "label": "Typical gap between orders",
        "risk_text": "Usually waits about {AvgGapDays:.0f} days between orders -- a long gap to bridge.",
        "safe_text": "Orders about every {AvgGapDays:.0f} days -- a regular rhythm.",
        "action": "Time outreach to their usual reorder cycle.",
    },
    "Orders_30d": {
        "concept": "recent_orders",
        "label": "Orders in the last 30 days",
        "risk_text": "Placed {Orders_30d:.0f} order(s) in the last 30 days -- quiet lately.",
        "safe_text": "Placed {Orders_30d:.0f} order(s) in the last 30 days.",
        "action": "Send a re-engagement offer while the relationship is still warm.",
    },
    "Orders_90d": {
        "concept": "recent_orders",
        "label": "Orders in the last 90 days",
        "risk_text": "Only {Orders_90d:.0f} order(s) in the last 90 days.",
        "safe_text": "{Orders_90d:.0f} order(s) in the last 90 days -- an active buyer.",
        "action": "Send a re-engagement offer while the relationship is still warm.",
    },
    "Orders_Prev30d": {
        "concept": "prior_orders",
        "label": "Orders in the 30 days before that",
        "risk_text": "Ordering pattern over the previous month points to a slowdown.",
        "safe_text": "Ordering pattern over the previous month looks healthy.",
        "action": None,
    },
    "Orders_Prev90d": {
        "concept": "prior_orders",
        "label": "Orders in the earlier 90 days",
        "risk_text": "Earlier ordering pattern points to a slowdown.",
        "safe_text": "Earlier ordering pattern looks healthy.",
        "action": None,
    },
    "Spend_90d": {
        "concept": "recent_spend",
        "label": "Spend in the last 90 days",
        "risk_text": "Spent only ₹{Spend_90d:,.0f} in the last 90 days.",
        "safe_text": "Spent ₹{Spend_90d:,.0f} in the last 90 days.",
        "action": "Highlight products related to past purchases to rebuild spend.",
    },
    "Spend_Prev90d": {
        "concept": "prior_spend",
        "label": "Spend in the earlier 90 days",
        "risk_text": "Earlier spending pattern points to a slowdown.",
        "safe_text": "Earlier spending pattern looks healthy.",
        "action": None,
    },
    "FreqTrendRatio": {
        "concept": "trend",
        "label": "Buying trend",
        "risk_text": "Order frequency is falling compared with the previous three months.",
        "safe_text": "Order frequency is steady or rising versus the previous three months.",
        "action": "Reach out before the slowdown turns into a lapse.",
    },
    "SpendTrendRatio": {
        "concept": "trend",
        "label": "Spending trend",
        "risk_text": "Spending is falling compared with the previous three months.",
        "safe_text": "Spending is steady or rising versus the previous three months.",
        "action": "Reach out before the slowdown turns into a lapse.",
    },
    "Orders_14d": {
        "concept": "recent_orders", "label": "Orders in the last 2 weeks",
        "risk_text": "No or few orders in the last 2 weeks ({Orders_14d:.0f}).",
        "safe_text": "Ordered {Orders_14d:.0f} time(s) in the last 2 weeks.",
        "action": "Send a re-engagement offer while the relationship is still warm.",
    },
    "Orders_180d": {
        "concept": "recent_orders", "label": "Orders in the last 6 months",
        "risk_text": "Only {Orders_180d:.0f} order(s) in the last 6 months.",
        "safe_text": "{Orders_180d:.0f} order(s) in the last 6 months -- a regular buyer.",
        "action": "Send a re-engagement offer while the relationship is still warm.",
    },
    "Spend_180d": {
        "concept": "recent_spend", "label": "Spend in the last 6 months",
        "risk_text": "Spent only ₹{Spend_180d:,.0f} in the last 6 months.",
        "safe_text": "Spent ₹{Spend_180d:,.0f} in the last 6 months.",
        "action": "Highlight products related to past purchases to rebuild spend.",
    },
    "RecentOrderShare": {
        "concept": "trend", "label": "Share of orders that are recent",
        "risk_text": "Most of their buying happened a while ago; recent activity is a small share.",
        "safe_text": "A large share of their buying is recent.",
        "action": "Reach out before the slowdown turns into a lapse.",
    },
    "RecentSpendShare": {
        "concept": "trend", "label": "Share of spend that is recent",
        "risk_text": "Most of their spend happened a while ago.",
        "safe_text": "A large share of their spend is recent.",
        "action": "Reach out before the slowdown turns into a lapse.",
    },
    "ActiveWeeks_90d": {
        "concept": "regularity", "label": "Weeks with an order (last 90 days)",
        "risk_text": "Ordered in only {ActiveWeeks_90d:.0f} different week(s) recently -- irregular.",
        "safe_text": "Ordered in {ActiveWeeks_90d:.0f} different weeks recently -- a steady habit.",
        "action": "Time outreach to their usual reorder cycle.",
    },
    "MedianGapDays": {
        "concept": "rhythm", "label": "Typical gap between orders",
        "risk_text": "Typically waits about {MedianGapDays:.0f} days between orders -- a long gap to bridge.",
        "safe_text": "Typically reorders every {MedianGapDays:.0f} days.",
        "action": "Time outreach to their usual reorder cycle.",
    },
    "GapStdDays": {
        "concept": "rhythm", "label": "How regular their ordering is",
        "risk_text": "Their gaps between orders vary a lot, so a quiet spell is harder to read.",
        "safe_text": "Orders at a fairly regular pace.",
        "action": None,
    },
    "LastGapDays": {
        "concept": "rhythm", "label": "Most recent gap between orders",
        "risk_text": "The gap before their latest order was long ({LastGapDays:.0f} days) -- the rhythm is stretching.",
        "safe_text": "Their latest order followed a normal gap ({LastGapDays:.0f} days).",
        "action": "Time outreach to their usual reorder cycle.",
    },
    "MaxGapDays": {
        "concept": "rhythm", "label": "Longest gap between orders",
        "risk_text": "Has gone as long as {MaxGapDays:.0f} days between orders before.",
        "safe_text": "Has never gone long between orders.",
        "action": None,
    },
    "RecencyOverMedianGap": {
        "concept": "overdue", "label": "Overdue vs. their usual rhythm",
        "risk_text": "Is {RecencyOverMedianGap:.1f}x overdue compared with their typical gap between orders.",
        "safe_text": "Still within their typical ordering rhythm.",
        "action": "Send a timely reminder or personalised offer -- they are past their usual reorder point.",
    },
    "OverdueZ": {
        "concept": "overdue", "label": "Overdue vs. their usual rhythm",
        "risk_text": "Has been silent longer than their own pattern would predict.",
        "safe_text": "Silence so far is within their own pattern.",
        "action": "Send a timely reminder or personalised offer -- they are past their usual reorder point.",
    },
    "LastOrderValue": {
        "concept": "last_order", "label": "Value of the latest order",
        "risk_text": "Their latest order (₹{LastOrderValue:,.0f}) was small.",
        "safe_text": "Their latest order was healthy (₹{LastOrderValue:,.0f}).",
        "action": "Suggest bundles or complementary products to lift basket size.",
    },
    "LastOrderRatio": {
        "concept": "last_order", "label": "Latest order vs. their usual",
        "risk_text": "Their latest order was smaller than usual.",
        "safe_text": "Their latest order was in line with or above usual.",
        "action": "Suggest bundles or complementary products to lift basket size.",
    },
    "AvgLinesPerOrder": {
        "concept": "basket", "label": "Items per order",
        "risk_text": "Typically buys few items per order ({AvgLinesPerOrder:.0f}).",
        "safe_text": "Buys many items per order ({AvgLinesPerOrder:.0f}) -- a deeply engaged basket.",
        "action": "Suggest bundles or complementary products to lift basket size.",
    },
    "RecentAOVRatio": {
        "concept": "basket", "label": "Recent order size vs. usual",
        "risk_text": "Recent orders are smaller than their usual.",
        "safe_text": "Recent orders are as large as usual or larger.",
        "action": "Suggest bundles or complementary products to lift basket size.",
    },
    "PAlive": {
        "concept": "alive", "label": "Likelihood still an active customer",
        "risk_text": "A statistical model of their buying pattern suggests they may already have drifted away.",
        "safe_text": "A statistical model of their buying pattern says they are very likely still active.",
        "action": "Send a timely reminder or personalised offer before they drift further.",
    },
    "ExpectedOrders": {
        "concept": "alive", "label": "Expected orders, next period",
        "risk_text": "Their buying pattern predicts few orders in the coming months ({ExpectedOrders:.1f}).",
        "safe_text": "Their buying pattern predicts about {ExpectedOrders:.1f} orders in the coming months.",
        "action": "Send a timely reminder or personalised offer before they drift further.",
    },
    "Age": {
        "concept": "age",
        "label": "Age",
        "risk_text": "Demographic profile (age {Age:.0f}) statistically correlates with higher churn in this dataset.",
        "safe_text": "Demographic profile (age {Age:.0f}) statistically correlates with lower churn in this dataset.",
        "action": None,
    },
    "Gender": {
        "concept": "gender",
        "label": "Gender",
        "risk_text": "Demographic profile statistically correlates with higher churn in this dataset.",
        "safe_text": "Demographic profile statistically correlates with lower churn in this dataset.",
        "action": None,
    },
}


# 2. PARSE A RAW SHAP FEATURE NAME


def base_feature_name(raw_name):
    """
    Turns a preprocessor-generated feature name into the original
    column name, e.g.:
        "num__EngagementScore"  -> "EngagementScore"
        "cat__Gender_Male"      -> "Gender"
    """
    name = raw_name.split("__", 1)[-1]

    # One-hot encoded categorical columns look like "Gender_Male" --
    # match against known categorical bases so we don't chop up a
    # numeric column name that happens to contain an underscore.
    for base in FEATURE_CONCEPTS:
        if name == base or name.startswith(base + "_"):
            return base
    return name


# 3. BUILD PLAIN-ENGLISH REASONS FOR ONE CUSTOMER


# Direction each feature is expected to point, used to keep reasons truthful.
# A linear model can give a feature an attribution that runs against common sense
# for one customer (correlated spend features offset each other), and the fixed
# wording of a reason ("spend is on the lower end") would then be false.
HIGHER_IS_SAFER = {
    "Frequency", "Monetary", "AverageOrderValue", "Orders_14d", "Orders_30d", "Orders_90d",
    "Orders_180d", "Spend_90d", "Spend_180d", "ActiveWeeks_90d", "ExpectedOrders", "PAlive",
    "AvgLinesPerOrder", "RecentOrderShare", "RecentSpendShare", "FreqTrendRatio",
    "SpendTrendRatio", "LastOrderValue", "LastOrderRatio", "RecentAOVRatio", "TenureDays",
}
HIGHER_IS_RISKIER = {
    "Recency", "RecencyOverGap", "RecencyOverMedianGap", "OverdueZ", "AvgGapDays",
    "MedianGapDays", "LastGapDays", "MaxGapDays",
}


def _contradicts_common_sense(base, is_risk, customer, reference):
    """True when a reason's wording would be false for this customer's actual value."""
    if not reference or base not in reference:
        return False
    try:
        value = float(customer.get(base))
        typical = float(reference[base])
    except (TypeError, ValueError):
        return False
    if value != value or typical != typical:
        return False
    if base in HIGHER_IS_SAFER:
        return (is_risk and value >= typical) or (not is_risk and value < typical)
    if base in HIGHER_IS_RISKIER:
        return (is_risk and value <= typical) or (not is_risk and value > typical)
    return False


def build_customer_reasons(customer, shap_rows, max_reasons=3, reference=None):
    """
    customer:   dict-like row of customer data (for filling in {value})
    shap_rows:  list of dicts with Feature / SHAP_Value for this one customer
    reference:  optional {feature: typical (median) value across customers}; when given,
                a reason whose wording contradicts the customer's real value is skipped.

    Returns up to `max_reasons` dicts:
        {label, text, direction ("risk"/"reassurance"), action, magnitude}
    Deduplicated so the same underlying concept isn't shown twice.
    """
    seen_concepts = set()
    reasons = []
    ranked_rows = sorted(shap_rows, key=lambda r: abs(float(r["SHAP_Value"])), reverse=True)
    for row in ranked_rows:
        base = base_feature_name(row["Feature"])
        concept_info = FEATURE_CONCEPTS.get(base)
        if concept_info is None:
            continue
        concept = concept_info["concept"]
        if concept in seen_concepts:
            continue
        is_risk = float(row["SHAP_Value"]) > 0
        if _contradicts_common_sense(base, is_risk, customer, reference):
            continue
        seen_concepts.add(concept)
        template = concept_info["risk_text"] if is_risk else concept_info["safe_text"]
        template = template.replace("\u20b9", CURRENCY_SYMBOL)
        try:
            text = template.format(**customer)
        except (KeyError, ValueError):
            text = template
        reasons.append(
            {
                "label": concept_info["label"],
                "text": text,
                "direction": "risk" if is_risk else "reassurance",
                "action": concept_info["action"] if is_risk else None,
                "magnitude": abs(float(row["SHAP_Value"])),
            }
        )
        if len(reasons) >= max_reasons:
            break
    return reasons
