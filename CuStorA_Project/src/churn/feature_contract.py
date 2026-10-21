"""
Which columns does the saved churn model actually expect?

Different training modes (a customer table vs. a transaction ledger) give
the model different input columns. Rather than every downstream stage
keeping its own hard-coded "drop these columns" list in sync, they all ask
the fitted model: a scikit-learn pipeline fitted on a DataFrame remembers
its input column names in `feature_names_in_`.
"""


def model_input_columns(model):
    """Column names the fitted model was trained on, or None if unknown."""
    names = getattr(model, "feature_names_in_", None)
    if names is None and hasattr(model, "named_steps"):
        names = getattr(model.named_steps.get("preprocessor"), "feature_names_in_", None)
    return list(names) if names is not None else None


def select_model_features(df, default_drop, model=None):
    """
    Return the frame to pass to `model`: exactly its training columns when
    they are all present, otherwise `df` minus `default_drop` (the old
    behaviour, kept for models that don't record their columns).
    """
    names = model_input_columns(model) if model is not None else None
    if names and all(name in df.columns for name in names):
        return df[names]
    return df.drop(columns=default_drop, errors="ignore")
