"""
Tests for app/valuation/ml_evaluation.py -- SHAP importance,
calibration curves, and feature-group ablations, all against the same
small synthetic dataset used in test_ml_valuation_classifier.py (kept
duplicated rather than imported across test files -- each test module
should be runnable/readable on its own).
"""

import numpy as np
import pandas as pd
import pytest

from app.valuation import ml_evaluation
from app.valuation.ml_features import FEATURE_COLUMNS
from app.valuation.ml_valuation_classifier import build_logreg, train_and_evaluate


def make_synthetic_training_df(n_per_label: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    labels = ["UNDERVALUED", "FAIRLY VALUED", "OVERVALUED"]
    rows = []
    for label in labels:
        offset = {"UNDERVALUED": 1.0, "FAIRLY VALUED": 0.0, "OVERVALUED": -1.0}[label]
        for _ in range(n_per_label):
            row = {col: float(rng.normal(loc=offset, scale=0.5)) for col in FEATURE_COLUMNS}
            row["realized_label"] = label
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_df():
    return make_synthetic_training_df()


@pytest.fixture
def fitted_logreg_and_split(synthetic_df):
    model = build_logreg()
    X = synthetic_df[FEATURE_COLUMNS]
    y = synthetic_df["realized_label"]
    model.fit(X, y)
    return model, X, y


def test_shap_importance_table_covers_every_feature_with_nonnegative_scores(fitted_logreg_and_split):
    model, X, _ = fitted_logreg_and_split

    table = ml_evaluation.shap_importance_table(model, X)

    assert sorted(table["feature"]) == sorted(FEATURE_COLUMNS)
    assert (table["importance"] >= 0).all()
    # sorted descending, as callers (the CLI script's printout) assume
    assert list(table["importance"]) == sorted(table["importance"], reverse=True)


def test_calibration_data_only_includes_labels_present_in_y_test(fitted_logreg_and_split):
    model, X, y = fitted_logreg_and_split

    result = ml_evaluation.calibration_data(model, X, y.to_numpy())

    assert set(result.keys()) <= {"UNDERVALUED", "FAIRLY VALUED", "OVERVALUED"}
    for label, points in result.items():
        assert len(points["mean_predicted"]) == len(points["fraction_positive"])
        assert all(0.0 <= p <= 1.0 for p in points["mean_predicted"])
        assert all(0.0 <= p <= 1.0 for p in points["fraction_positive"])


def test_calibration_data_drops_a_label_with_too_few_test_samples():
    # Only one FAIRLY VALUED row in the whole (tiny) set -- too few for
    # even 2 calibration bins, so it must be skipped rather than erroring.
    rng = np.random.default_rng(0)
    rows = []
    for label, n in [("UNDERVALUED", 6), ("OVERVALUED", 6), ("FAIRLY VALUED", 1)]:
        for _ in range(n):
            rows.append({**{c: float(rng.normal()) for c in FEATURE_COLUMNS}, "realized_label": label})
    df = pd.DataFrame(rows)
    model = build_logreg()
    model.fit(df[FEATURE_COLUMNS], df["realized_label"])

    result = ml_evaluation.calibration_data(model, df[FEATURE_COLUMNS], df["realized_label"].to_numpy())

    assert "FAIRLY VALUED" not in result


def test_run_feature_ablations_covers_every_group_with_a_delta_from_baseline(synthetic_df):
    result = ml_evaluation.run_feature_ablations(synthetic_df, best_model_name="logistic_regression")

    assert 0.0 <= result["baseline_macro_f1"] <= 1.0
    assert set(result["ablations"]) == set(ml_evaluation.FEATURE_GROUPS)
    for group_name, group_result in result["ablations"].items():
        assert group_result["removed_features"] == ml_evaluation.FEATURE_GROUPS[group_name]
        expected_delta = group_result["macro_f1_without_group"] - result["baseline_macro_f1"]
        assert group_result["macro_f1_delta"] == pytest.approx(expected_delta)


def test_run_feature_ablations_never_touches_the_shared_feature_columns_list(synthetic_df):
    # FEATURE_COLUMNS is imported module-level state shared with
    # ml_features.py/ml_valuation_classifier.py -- an ablation run
    # mutating it in place would corrupt every other caller for the
    # rest of the process, not just this one test.
    before = list(FEATURE_COLUMNS)

    ml_evaluation.run_feature_ablations(synthetic_df, best_model_name="logistic_regression")

    assert FEATURE_COLUMNS == before
