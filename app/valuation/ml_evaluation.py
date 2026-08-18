"""
ml_evaluation.py

Deeper evaluation for the ML valuation classifier
(ml_valuation_classifier.py): SHAP-based feature importance, per-class
calibration curves, and feature-group ablations. Kept out of
ml_valuation_classifier.py deliberately -- that module's train()/
predict_verdict() sit on the always-imported live-inference path
(every research job that reaches ValuationTool touches predict_verdict()),
so an optional, heavier dependency (shap) failing to import can never
break live inference. Only scripts/train_ml_classifier.py imports this
module, on-demand.

SHAP over the plain coefficient/gain importance already in
feature_importance_table(): that function uses a different notion of
"importance" per model family (raw logistic-regression coefficient
magnitude vs. tree gain), which aren't on the same scale and can't be
honestly compared across the two models being evaluated. SHAP values
are model-agnostic (mean |SHAP value| per feature, from each model's
own predict_proba), so the same ranking method applies to both.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from app.valuation.ml_features import FEATURE_COLUMNS
from app.valuation.ml_valuation_classifier import LABELS, build_logreg, build_secondary_model

# Named groups of FEATURE_COLUMNS for ablation studies -- which
# *category* of signal actually moves held-out performance, not just
# which single column. Grouped by where the number comes from in the
# valuation pipeline (see ml_features.extract_features).
FEATURE_GROUPS: Dict[str, List[str]] = {
    "dcf": ["growth_rate", "wacc", "dcf_over_price"],
    "monte_carlo": ["mc_mean_over_price", "mc_std_over_price", "mc_prob_undervalued"],
    "balance_sheet": ["net_cash_per_share_over_price"],
    "market_relative": ["relative_vs_history_pct", "beta"],
    "growth_yield": ["revenue_growth", "fcf_yield"],
}


def shap_importance_table(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """
    Model-agnostic SHAP importance: mean |SHAP value| per feature,
    averaged across samples and classes. Uses shap.Explainer's generic
    predict_proba path (works identically for the sklearn Pipeline and
    the tree model) rather than TreeExplainer/LinearExplainer
    special-casing -- this dataset is tens of rows, so the slower
    generic path's runtime is a non-issue, and one code path mattering
    more than optimizing an explainer that runs once per training run.
    """
    import shap

    explainer = shap.Explainer(model.predict_proba, X)
    shap_values = explainer(X)

    values = shap_values.values
    # (n_samples, n_features, n_classes) for multiclass predict_proba;
    # (n_samples, n_features) if shap ever collapses to one output --
    # mean |.| over every axis except features either way.
    axes = tuple(i for i in range(values.ndim) if i != 1)
    importances = np.abs(values).mean(axis=axes)

    return pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": importances}
    ).sort_values("importance", ascending=False).reset_index(drop=True)


def calibration_data(model: Any, X_test: pd.DataFrame, y_test: np.ndarray, n_bins: int = 5) -> Dict[str, Dict[str, List[float]]]:
    """
    Per-class reliability diagram data (one-vs-rest): for each label,
    (mean predicted probability, actual fraction positive) pairs across
    probability buckets. A well-calibrated model's points sit near the
    diagonal -- a batch of "70%-confidence UNDERVALUED" calls should be
    right about 70% of the time, not just directionally more often than
    not. Silently drops a label if there are too few held-out samples
    for even 2 bins to be meaningful, rather than raising -- this
    dataset is small enough that some labels can be scarce in any
    given test split.
    """
    proba = model.predict_proba(X_test)
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model.named_steps["clf"].classes_)

    result: Dict[str, Dict[str, List[float]]] = {}
    for label in LABELS:
        if label not in classes:
            continue
        class_idx = classes.index(label)
        y_binary = (np.asarray(y_test) == label).astype(int)
        # min(), not max(), of the two class counts -- a meaningful
        # reliability curve for this label needs enough of BOTH the
        # positive and negative one-vs-rest examples; gating on
        # whichever count is larger (almost always the negatives)
        # let single-digit-positive cases through with noise-only bins.
        effective_bins = min(n_bins, y_binary.sum(), len(y_binary) - y_binary.sum())
        if effective_bins < 2:
            continue
        fraction_pos, mean_pred = calibration_curve(
            y_binary, proba[:, class_idx], n_bins=effective_bins, strategy="quantile"
        )
        result[label] = {
            "mean_predicted": mean_pred.tolist(),
            "fraction_positive": fraction_pos.tolist(),
        }
    return result


def run_feature_ablations(df: pd.DataFrame, best_model_name: str, test_size: float = 0.25) -> Dict[str, Any]:
    """
    Retrains the best model type once per FEATURE_GROUPS entry with
    that group's columns removed, comparing held-out macro-F1 against
    the full-feature baseline (same split logic as
    ml_valuation_classifier.train_and_evaluate, but not reusing its
    single fixed split -- each ablation needs its own X of a different
    width). Answers "does this category of signal actually earn its
    place" instead of assuming it does because it's in FEATURE_COLUMNS.
    """
    y = df["realized_label"].astype(str).to_numpy()
    class_counts = np.unique(y, return_counts=True)[1]
    stratify = y if class_counts.min() >= 2 else None

    def _macro_f1(columns: List[str]) -> float:
        X = df[columns].astype("float64").to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        model = build_logreg() if best_model_name == "logistic_regression" else build_secondary_model()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return float(f1_score(y_test, preds, average="macro", zero_division=0))

    baseline_f1 = _macro_f1(FEATURE_COLUMNS)

    ablations = {}
    for group_name, group_cols in FEATURE_GROUPS.items():
        remaining = [c for c in FEATURE_COLUMNS if c not in group_cols]
        if not remaining:
            continue
        f1_without = _macro_f1(remaining)
        ablations[group_name] = {
            "removed_features": group_cols,
            "macro_f1_without_group": f1_without,
            "macro_f1_delta": f1_without - baseline_f1,
        }

    return {"baseline_macro_f1": baseline_f1, "ablations": ablations}
