"""
Maps an arbitrary uploaded customer dataset onto CuStorA's canonical
schema (see app.py's REQUIRED_COLUMNS), instead of requiring an
uploaded file to already use CuStorA's exact column names.

This is deliberately scoped down from a full "detect -> suggest ->
user confirms -> handle missing fields" UI flow (project review item
#6). It's the rule-based matching core that flow would sit on top of,
built and validated against one genuinely external dataset (see
scripts/run_external_validation.py) rather than designed from theory
first -- per the project review's phase ordering: "build column
mapping around real problems," not before seeing any.

Two kinds of "match" are treated differently on purpose:

- STRUCTURAL match: the source column is literally the same
  quantity, just named differently or in a different unit/format
  (e.g. "customer_id" -> CustomerID, "TotalCharges" -> TotalSpent).
  These are auto-mapped.
- NO match: nothing in the source data measures this canonical field
  at all (e.g. no dataset without a return-tracking system can supply
  ReturnRate). These are reported as unavailable, not guessed at or
  filled with a placeholder -- see clean_dataset()'s "don't invent
  missing business data" principle, which this module extends to the
  mapping step itself.

A column is never force-mapped just because the names are similar if
the underlying meaning is actually different (e.g. a telecom
"tenure in months" column is NOT mapped to CuStorA's TotalPurchases,
even though both are integers about customer activity -- one counts
elapsed time, the other counts discrete purchase events).
"""

import re

import pandas as pd

# CuStorA's canonical schema (mirrors app.py REQUIRED_COLUMNS, plus a
# short human description of what each field actually measures, used
# in the coverage report).
CANONICAL_SCHEMA = {
    "CustomerID": "Unique identifier for the customer",
    "Age": "Customer's age in years",
    "Gender": "Customer's gender/category label",
    "RegistrationDate": "Date the customer relationship started",
    "LastPurchaseDate": "Date of the customer's most recent purchase/order",
    "TotalPurchases": "Count of discrete purchase/order events, all-time",
    "TotalSpent": "Total revenue from the customer, all-time",
    "AverageOrderValue": "Average revenue per discrete purchase/order",
    "EngagementScore": "A 0-100 behavioural engagement score",
    "Complaints": "Count of service complaints, all-time",
    "Returns": "Count of product/order returns, all-time",
    "DiscountUsage": "A 0-100 measure of discount/promo reliance",
    "Churn": "Whether the customer has churned (1) or not (0)",
}

# Synonyms are intentionally narrow -- each one is a genuine
# structural match, not a loose guess. A source column not covered
# here is left unmapped rather than fuzzy-matched into something
# that isn't actually the same quantity.
SYNONYMS = {
    "CustomerID": ["customerid", "customer_id", "custid", "id", "clientid", "userid"],
    "Age": ["age", "customerage"],
    "Gender": ["gender", "sex"],
    "RegistrationDate": [
        "registrationdate", "signupdate", "joindate", "startdate",
        "customersince", "accountcreated", "registration_date",
    ],
    "LastPurchaseDate": [
        "lastpurchasedate", "lastorderdate", "lastactivity", "lastactivitydate",
        "last_purchase_date", "mostrecentorder",
    ],
    "TotalPurchases": [
        "totalpurchases", "orders", "numorders", "purchasecount", "ordercount",
        "totalorders", "numberoforders",
    ],
    "TotalSpent": [
        "totalspent", "totalrevenue", "totalcharges", "lifetimerevenue",
        "totalvalue", "revenue",
    ],
    "AverageOrderValue": ["averageordervalue", "aov", "avgordervalue", "meanordervalue"],
    "EngagementScore": ["engagementscore", "engagement"],
    "Complaints": ["complaints", "numcomplaints", "complaintcount", "supportcontacts"],
    "Returns": ["returns", "numreturns", "returncount"],
    "DiscountUsage": ["discountusage", "discount", "discountrate", "promousage"],
    "Churn": ["churn", "churned", "churnvalue", "churnlabel", "attrition", "ischurned"],
}


def _normalize(name):
    """Lowercase and strip everything that isn't a letter or digit,
    so "Customer ID", "customer_id", and "CustomerID" all normalize
    to the same token for comparison."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def suggest_mapping(source_columns):
    """
    Compare `source_columns` (an uploaded file's column names) against
    CANONICAL_SCHEMA and return a MappingResult-shaped dict:

        {
            "mapped": {canonical_name: source_column, ...},
            "unmapped_canonical": [canonical_name, ...],   # required fields with no match
            "unmapped_source": [source_column, ...],       # source columns not used
        }

    Matching is exact-after-normalization first (handles case,
    spacing, punctuation), then checked against SYNONYMS. No fuzzy
    string-distance matching is used deliberately -- a near-miss
    string match (e.g. "Recency" vs "Currency") being auto-accepted
    is worse than leaving a field correctly flagged as unmapped.
    """
    normalized_source = {_normalize(col): col for col in source_columns}
    mapped = {}
    used_source_columns = set()
    for canonical_name in CANONICAL_SCHEMA:
        candidates = [_normalize(canonical_name)] + [
            _normalize(s) for s in SYNONYMS.get(canonical_name, [])
        ]
        for candidate in candidates:
            if candidate in normalized_source:
                source_col = normalized_source[candidate]
                mapped[canonical_name] = source_col
                used_source_columns.add(source_col)
                break
    unmapped_canonical = [c for c in CANONICAL_SCHEMA if c not in mapped]
    unmapped_source = [c for c in source_columns if c not in used_source_columns]
    return {
        "mapped": mapped,
        "unmapped_canonical": unmapped_canonical,
        "unmapped_source": unmapped_source,
    }


def build_coverage_report(mapping_result, dataset_name="uploaded dataset"):
    """Turn a suggest_mapping() result into a presentable DataFrame:
    one row per canonical field, showing whether it mapped and what
    it means if it didn't -- this is the artifact
    scripts/run_external_validation.py and (eventually) app.py's
    upload flow show to a person, not the raw dict."""
    rows = []
    for canonical_name, description in CANONICAL_SCHEMA.items():
        if canonical_name in mapping_result["mapped"]:
            rows.append({
                "CuStorA Field": canonical_name,
                "Meaning": description,
                "Status": "Mapped",
                "Source Column": mapping_result["mapped"][canonical_name],
            })
        else:
            rows.append({
                "CuStorA Field": canonical_name,
                "Meaning": description,
                "Status": "Unavailable",
                "Source Column": "",
            })
    report = pd.DataFrame(rows)
    mapped_count = (report["Status"] == "Mapped").sum()
    print(f"\nColumn mapping for {dataset_name}: "
          f"{mapped_count}/{len(CANONICAL_SCHEMA)} CuStorA fields mapped.")
    return report


def apply_mapping(source_df, mapping_result):
    """Build a DataFrame using CuStorA's canonical column names,
    filled only where a real mapping exists. Unmapped canonical
    columns are left entirely absent from the result (NOT filled with
    NaN-as-a-placeholder-that-looks-like-real-missing-data) -- the
    caller is responsible for deciding what to do about a canonical
    field with no source column at all, which is a different
    situation from "the column exists but this row's value is
    missing" that clean_dataset() already handles."""
    result = pd.DataFrame(index=source_df.index)
    for canonical_name, source_col in mapping_result["mapped"].items():
        result[canonical_name] = source_df[source_col]
    return result
