"""
"What happened to my data in this stage?" -- for the analysis status page.

Everything here is READ from files the pipeline already wrote (the per-stage
checkpoints in data/processed/ and the summaries in reports/). Nothing is
recomputed or invented: if a file is missing the stage simply says so.

describe_stage(name, root, currency) returns a plain dict:
    headline : one sentence on what this stage did to the data
    stats    : [{"label", "value"}]          small number tiles
    bars     : [{"label", "text", "share"}]  a distribution (share is 0-100)
    chips    : [str]                         short names (e.g. new columns)
    note     : optional extra sentence
    rows / seconds / file : where the numbers come from
"""

import json
from pathlib import Path

import pandas as pd

PROCESSED = "data/processed"

# The checkpoint each stage leaves behind (None = the stage's output is a model / report).
STAGE_FILES = {
    "Cleaning": f"{PROCESSED}/customers_cleaned.csv",
    "Feature Engineering": f"{PROCESSED}/customers_features.csv",
    "RFM Analysis": f"{PROCESSED}/customers_rfm.csv",
    "PCA": f"{PROCESSED}/customers_pca.csv",
    "K-Means Clustering": f"{PROCESSED}/customers_clustered.csv",
    "Segment Naming": f"{PROCESSED}/customers_segmented.csv",
    "Churn Prediction": "models/churn/best_churn_model.joblib",
    "Relationship Health": f"{PROCESSED}/customers_health.csv",
    "SHAP Explainability": "reports/shap/global_feature_importance.csv",
    "Customer State": f"{PROCESSED}/customers_state.csv",
    "Retention Recommendations": f"{PROCESSED}/customers_recommendations.csv",
    "CLV and Business Metrics": f"{PROCESSED}/customers_business.csv",
}


# ---------------------------------------------------------------- small helpers


def _csv(root, rel, columns=None):
    """Read a CSV (only the wanted columns); None if it is missing or unreadable."""
    path = Path(root) / rel
    try:
        if columns is None:
            return pd.read_csv(path)
        wanted = set(columns)
        return pd.read_csv(path, usecols=lambda c: c in wanted)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None


def _json(root, rel):
    try:
        return json.loads((Path(root) / rel).read_text())
    except (OSError, ValueError):
        return None


def _n(value):
    return f"{int(round(float(value))):,}"


def _money(value, currency):
    v, a = float(value), abs(float(value))
    sign = "-" if v < 0 else ""
    if currency == "\u20b9":
        if a >= 1e7:
            return f"{sign}{currency}{a / 1e7:.2f}Cr"
        if a >= 1e5:
            return f"{sign}{currency}{a / 1e5:.1f}L"
    else:
        if a >= 1e6:
            return f"{sign}{currency}{a / 1e6:.2f}M"
        if a >= 1e4:
            return f"{sign}{currency}{a / 1e3:.1f}K"
    return f"{sign}{currency}{a:,.0f}"


def _stat(label, value):
    return {"label": label, "value": str(value)}


def _bars(pairs, total=None, limit=8):
    """pairs: [(label, count)] -> bars sized by share of the total."""
    pairs = [(str(k), float(v)) for k, v in pairs][:limit]
    total = float(total) if total else (sum(v for _, v in pairs) or 1.0)
    return [
        {"label": k, "text": f"{_n(v)} ({v / total * 100:.0f}%)", "share": round(v / total * 100, 1)}
        for k, v in pairs
    ]


def _counts(series, limit=8):
    vc = series.dropna().value_counts()
    return _bars(vc.items(), total=int(vc.sum()), limit=limit)


# ---------------------------------------------------------------- one builder per stage


def _cleaning(root, cur):
    health = _csv(root, "reports/data_health.csv")
    if health is None or health.empty:
        return None
    values = dict(zip(health["Metric"], health["Value"]))
    before, after = values.get("Rows before cleaning"), values.get("Rows after cleaning")
    out = {"stats": [_stat(str(k), v) for k, v in list(values.items())[:12]]}
    if before is not None and after is not None:
        out["headline"] = f"{_n(before)} customers went in and {_n(after)} came out after cleaning."
    else:
        out["headline"] = "Your data was checked for duplicates, missing values and impossible numbers."
    coverage = _csv(root, "reports/data_coverage.csv")
    if coverage is not None and "Status" in coverage:
        out["bars"] = _counts(coverage["Status"].astype(str).str.strip().replace({"Not": "Not available"}))
        out["note"] = (
            "Bars: how each field was obtained. Observed = in your file, Derived = calculated from it, "
            "Not available = missing from your file (left out, never guessed)."
        )
    return out


def _features(root, cur):
    feat = _csv(root, "data/processed/customers_features.csv")
    cleaned = _csv(root, "data/processed/customers_cleaned.csv")
    if feat is None:
        return None
    new = [c for c in feat.columns if cleaned is None or c not in cleaned.columns]
    out = {
        "headline": f"{len(new)} new columns were created from your cleaned data, such as days since last purchase.",
        "chips": new[:14],
        "stats": [_stat("New columns", len(new)), _stat("Columns now", len(feat.columns))],
    }
    for col, label, fmt in (
        ("Recency", "Typical days since last purchase", lambda v: f"{v:,.0f}"),
        ("Frequency", "Typical number of orders", lambda v: f"{v:,.0f}"),
        ("Monetary", "Typical total spend", lambda v: _money(v, cur)),
    ):
        if col in feat.columns:
            out["stats"].append(_stat(label, fmt(pd.to_numeric(feat[col], errors="coerce").median())))
    out["note"] = "Typical values are medians across all customers."
    return out


def _rfm(root, cur):
    d = _csv(root, "data/processed/customers_rfm.csv", ["R_Score", "F_Score", "M_Score", "RFM_Segment", "RFM_Total"])
    if d is None or "RFM_Segment" not in d:
        return None
    out = {
        "headline": "Every customer got a 1 to 5 score for how recently, how often and how much they buy.",
        "bars": _counts(d["RFM_Segment"]),
        "stats": [],
    }
    for col, label in (("R_Score", "Average recency score"), ("F_Score", "Average frequency score"), ("M_Score", "Average spend score")):
        if col in d:
            out["stats"].append(_stat(label, f"{d[col].mean():.1f}"))
    out["note"] = "Bars: the classic RFM groups your customers fell into."
    return out


def _pca(root, cur):
    d = _csv(root, "data/processed/customers_pca.csv", ["PC1", "PC2"])
    if d is None or "PC1" not in d:
        return None
    return {
        "headline": "Each customer was given a spot on a 2-D map (PC1 and PC2) that keeps customers with similar behaviour close together.",
        "stats": [
            _stat("Columns added", 2),
            _stat("PC1 range", f"{d['PC1'].min():.1f} to {d['PC1'].max():.1f}"),
            _stat("PC2 range", f"{d['PC2'].min():.1f} to {d['PC2'].max():.1f}"),
        ],
        "note": "This map is only for looking at your customers; the groups themselves are found in the next stage. See it on the Customer clusters page.",
    }


def _kmeans(root, cur):
    d = _csv(root, "data/processed/customers_clustered.csv", ["Cluster"])
    if d is None or "Cluster" not in d:
        return None
    k = int(d["Cluster"].nunique())
    out = {
        "headline": f"Customers were split into {k} groups that behave alike.",
        "bars": _bars([(f"Cluster {c}", n) for c, n in d["Cluster"].value_counts().sort_index().items()], total=len(d)),
        "stats": [_stat("Groups found", k)],
    }
    ev = _csv(root, "reports/kmeans_evaluation.csv")
    if ev is not None and "K" in ev:
        out["stats"].append(_stat("Group counts tried", f"{int(ev['K'].min())} to {int(ev['K'].max())}"))
        row = ev[ev["K"] == k]
        if not row.empty and "Silhouette" in row:
            out["stats"].append(_stat("Separation score (silhouette)", f"{float(row['Silhouette'].iloc[0]):.2f}"))
    return out


def _naming(root, cur):
    d = _csv(root, "data/processed/customers_segmented.csv", ["SegmentName"])
    if d is None or "SegmentName" not in d:
        return None
    return {
        "headline": f"The groups were given {d['SegmentName'].nunique()} plain-English names your team can act on.",
        "bars": _counts(d["SegmentName"]),
    }


def _churn(root, cur):
    comparison = _csv(root, "reports/churn/churn_model_comparison.csv")
    thresholds = _json(root, "reports/churn/decision_thresholds.json") or {}
    meta = _json(root, "reports/run_meta.json") or {}
    if comparison is None and not thresholds:
        return None
    out = {"stats": [], "bars": []}
    selected = None
    if comparison is not None and "Model" in comparison:
        if "Selected" in comparison:
            chosen = comparison[comparison["Selected"].astype(str).str.lower() == "true"]
            selected = chosen["Model"].iloc[0] if not chosen.empty else None
        out["stats"].append(_stat("Models compared", len(comparison)))
        score = next((c for c in comparison.columns if "ROC" in c), None)
        if score:
            out["bars"] = [
                {"label": str(r["Model"]) + (" (chosen)" if r["Model"] == selected else ""),
                 "text": f"{float(r[score]):.3f}", "share": round(float(r[score]) * 100, 1)}
                for _, r in comparison.iterrows()
            ]
    out["headline"] = (
        f"Several models learned from past customers; “{selected}” was chosen and now scores everyone."
        if selected else "A churn model scored every customer's chance of leaving."
    )
    auc = meta.get("test_roc_auc")
    if auc is None:
        final = _csv(root, "reports/churn/final_test_evaluation.csv")
        if final is not None and "ROC-AUC" in final and not final.empty:
            auc = round(float(final["ROC-AUC"].iloc[0]), 3)
    if auc is not None:
        out["stats"].append(_stat("Accuracy on unseen data (ROC-AUC)", auc))
    if "medium_risk_threshold" in thresholds:
        out["stats"].append(_stat("“Likely to leave” line", f"{float(thresholds['medium_risk_threshold']) * 100:.0f}% chance"))
    out["note"] = "Bars: each model's score on a validation period (higher is better; 0.5 is a coin flip)."
    return out


def _health(root, cur):
    d = _csv(root, "data/processed/customers_health.csv", ["RelationshipHealth", "RelationshipHealthScore"])
    if d is None or "RelationshipHealth" not in d:
        return None
    out = {
        "headline": "Every customer got a relationship health score out of 100, separate from their churn chance.",
        "bars": _counts(d["RelationshipHealth"]),
        "stats": [],
    }
    if "RelationshipHealthScore" in d:
        out["stats"].append(_stat("Average health score", f"{d['RelationshipHealthScore'].mean():.0f} / 100"))
    return out


def _shap(root, cur):
    # Imported here so this module stays cheap to import and easy to test.
    from src.explainability import reason_translator
    from src.explainability.reason_translator import base_feature_name

    imp = _csv(root, "reports/shap/global_feature_importance.csv")
    if imp is None or imp.empty:
        return None
    drivers = {}
    for _, row in imp.iterrows():
        base = base_feature_name(row["Feature"])
        label = reason_translator.FEATURE_CONCEPTS.get(base, {}).get("label", base)
        drivers[label] = drivers.get(label, 0.0) + float(row["MeanAbsoluteSHAP"])
    ranked = sorted(drivers.items(), key=lambda kv: kv[1], reverse=True)[:6]
    top = ranked[0][1] or 1.0
    return {
        "headline": "The model's predictions were traced back to the behaviours that push risk up or down the most.",
        "bars": [{"label": k, "text": f"{v / top * 100:.0f}%", "share": round(v / top * 100, 1)} for k, v in ranked],
        "note": "Bars: influence relative to the strongest behaviour. Each customer's own reasons appear when you open them.",
    }


def _state(root, cur):
    s = _csv(root, "reports/customer_state_summary.csv")
    if s is None or s.empty:
        return None
    s = s.sort_values("CustomerCount", ascending=False)
    return {
        "headline": "Churn chance and relationship health were combined into one customer state.",
        "bars": _bars(zip(s["CustomerState"], s["CustomerCount"]), total=s["CustomerCount"].sum()),
    }


def _recommend(root, cur):
    s = _csv(root, "reports/retention_recommendation_summary.csv")
    if s is None or s.empty:
        return None
    s = s.sort_values("CustomerCount", ascending=False)
    return {
        "headline": "Every customer was matched to one suggested action for your team.",
        "bars": _bars(zip(s["Recommendation"], s["CustomerCount"]), total=s["CustomerCount"].sum()),
    }


def _business(root, cur):
    d = _csv(root, "data/processed/customers_business.csv", ["BusinessPriority", "RevenueAtStake", "ExpectedLoss"])
    if d is None or "BusinessPriority" not in d:
        return None
    out = {
        "headline": "Each customer's money at stake was worked out, and customers were ranked by who needs attention first.",
        "bars": _counts(d["BusinessPriority"]),
        "stats": [],
    }
    if "RevenueAtStake" in d:
        out["stats"].append(_stat("Revenue at stake", _money(d["RevenueAtStake"].sum(), cur)))
    if "ExpectedLoss" in d:
        out["stats"].append(_stat("Expected loss", _money(d["ExpectedLoss"].sum(), cur)))
    return out


BUILDERS = {
    "Cleaning": _cleaning,
    "Feature Engineering": _features,
    "RFM Analysis": _rfm,
    "PCA": _pca,
    "K-Means Clustering": _kmeans,
    "Segment Naming": _naming,
    "Churn Prediction": _churn,
    "Relationship Health": _health,
    "SHAP Explainability": _shap,
    "Customer State": _state,
    "Retention Recommendations": _recommend,
    "CLV and Business Metrics": _business,
}


def describe_stage(stage, root=".", currency="\u00a3"):
    """What this stage did to the user's data, read from the files the run left behind."""
    builder = BUILDERS.get(stage)
    result = None
    if builder is not None:
        try:
            result = builder(root, currency)
        except Exception:  # a detail panel must never take the page down
            result = None
    if result is None:
        return {
            "stage": stage,
            "available": False,
            "headline": "Details for this stage are not available (its output file was not found).",
            "stats": [], "bars": [], "chips": [],
        }
    result.setdefault("stats", [])
    result.setdefault("bars", [])
    result.setdefault("chips", [])
    result.update(stage=stage, available=True, file=STAGE_FILES.get(stage))

    # Customers in this stage's checkpoint, and how long the stage took.
    rel = STAGE_FILES.get(stage, "")
    if rel.endswith(".csv") and rel.startswith(PROCESSED):
        ids = _csv(root, rel, ["CustomerID"])
        if ids is not None:
            result["rows"] = int(len(ids))
    status = _csv(root, "reports/pipeline_status.csv")
    if status is not None and "Stage" in status:
        row = status[status["Stage"] == stage]
        if not row.empty and pd.notna(row["Seconds"].iloc[0]):
            result["seconds"] = float(row["Seconds"].iloc[0])
    return result
