"""
Unit tests for ml_valuation_classifier.py (app/valuation/).

A real trained model artifact IS committed in this checkout
(app/valuation/valuation_classifier.joblib, tracked in git -- confirmed
via `git ls-files`), so predict_verdict() is exercised against the
real saved model rather than a mock. Training-pipeline tests
(cross_validate_models/train_and_evaluate/train) use a small synthetic,
separable feature set and monkeypatch MODEL_PATH/METRICS_PATH to a
tmp_path so they never overwrite the real committed artifact.
"""

import os

import numpy as np
import pandas as pd
import pytest

from app.valuation import ml_valuation_classifier as mlc
from app.valuation.ml_valuation_classifier import (
    FEATURE_COLUMNS, LABELS, MODEL_PATH, load_training_data, predict_verdict,
)


# ---------------------------------------------------------------- predict_verdict (real saved model)

def test_predict_verdict_returns_a_verdict_and_probabilities_for_a_valid_feature_row():
    row = {c: 0.1 for c in FEATURE_COLUMNS}
    result = predict_verdict(row)

    assert result is not None
    assert result["verdict"] in LABELS
    assert set(result["probabilities"].keys()) == set(LABELS)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= p <= 1.0 for p in result["probabilities"].values())
    assert result["model_name"] in ("logistic_regression", mlc._secondary_model_name())


def test_predict_verdict_returns_none_when_a_required_feature_is_missing():
    row = {c: 0.1 for c in FEATURE_COLUMNS}
    del row[FEATURE_COLUMNS[0]]
    assert predict_verdict(row) is None



def test_predict_verdict_returns_none_when_a_required_feature_is_explicitly_none():
    row = {c: 0.1 for c in FEATURE_COLUMNS}
    row[FEATURE_COLUMNS[-1]] = None
    assert predict_verdict(row) is None


def test_predict_verdict_returns_none_when_no_model_file_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(mlc, "MODEL_PATH", str(tmp_path / "nonexistent.joblib"))
    row = {c: 0.1 for c in FEATURE_COLUMNS}
    assert predict_verdict(row) is None


def test_predict_verdict_raises_on_a_non_numeric_feature_value():
    # CURRENT actual behavior, documented rather than silently "fixed":
    # predict_verdict only guards against a MISSING feature (None/absent
    # via `feature_row.get(c) is None`), not a malformed one. A
    # non-numeric value reaches pd.DataFrame -> StandardScaler and
    # raises ValueError instead of degrading to None the way the
    # missing-feature case does. Reported upstream as a real gap; not
    # fixed here per instructions to not modify source modules.
    row = {c: 0.1 for c in FEATURE_COLUMNS}
    row[FEATURE_COLUMNS[0]] = "not_a_number"
    with pytest.raises(ValueError):
        predict_verdict(row)


def test_predict_verdict_raises_on_a_nan_feature_value():
    # Same gap as above, via a different malformed input: a NaN isn't
    # caught by the `is None` check either (NaN is not None), and
    # sklearn's LogisticRegression rejects NaN input outright.
    row = {c: 0.1 for c in FEATURE_COLUMNS}
    row[FEATURE_COLUMNS[0]] = float("nan")
    with pytest.raises(ValueError):
        predict_verdict(row)


def test_model_path_points_at_a_committed_artifact_in_this_checkout():
    import os
    assert os.path.exists(MODEL_PATH)


# ---------------------------------------------------------------- load_training_data

def test_load_training_data_raises_on_missing_feature_columns(tmp_path):
    csv_path = tmp_path / "training.csv"
    df = pd.DataFrame({"growth_rate": [0.1, 0.2], "realized_label": ["UNDERVALUED", "OVERVALUED"]})
    df.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="missing feature columns"):
        load_training_data(str(csv_path))


def test_load_training_data_drops_rows_with_missing_features_or_label(tmp_path):
    csv_path = tmp_path / "training.csv"
    rows = []
    for i in range(3):
        row = {c: 0.1 * i for c in FEATURE_COLUMNS}
        row["realized_label"] = "FAIRLY VALUED"
        rows.append(row)
    # A row with a missing feature value must be dropped.
    incomplete = {c: 0.1 for c in FEATURE_COLUMNS}
    incomplete[FEATURE_COLUMNS[0]] = None
    incomplete["realized_label"] = "OVERVALUED"
    rows.append(incomplete)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = load_training_data(str(csv_path))
    assert len(df) == 3
    assert df["realized_label"].tolist() == ["FAIRLY VALUED"] * 3


# ---------------------------------------------------------------- training pipeline (synthetic, tmp-path-isolated)

def _synthetic_training_df(rows_per_class=10, seed=0):
    """Small, deliberately-separable synthetic dataset -- feature
    values cluster around a distinct mean per label so the pipeline
    exercises real, non-degenerate model fitting rather than pure
    noise, without needing real financial data."""
    rng = np.random.default_rng(seed)
    rows = []
    for class_idx, label in enumerate(LABELS):
        for _ in range(rows_per_class):
            row = {c: float(rng.normal(loc=class_idx * 3.0, scale=0.2)) for c in FEATURE_COLUMNS}
            row["realized_label"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def test_cross_validate_models_returns_metrics_for_both_candidate_models():
    df = _synthetic_training_df(rows_per_class=10)
    results = mlc.cross_validate_models(df)

    assert set(results.keys()) == {"logistic_regression", mlc._secondary_model_name()}
    for name, metrics in results.items():
        assert 0.0 <= metrics["cv_accuracy_mean"] <= 1.0
        assert 0.0 <= metrics["cv_f1_macro_mean"] <= 1.0
        assert metrics["n_splits"] >= 2


def test_cross_validate_models_caps_n_splits_at_the_smallest_class_count():
    # 3 rows in the smallest class -> n_splits must be clamped to 3,
    # not the requested default of 5 (StratifiedKFold can't have more
    # folds than the smallest class has members).
    df = _synthetic_training_df(rows_per_class=10)
    sparse_label = LABELS[0]
    keep = df[df["realized_label"] != sparse_label].copy()
    sparse_rows = df[df["realized_label"] == sparse_label].iloc[:3]
    df = pd.concat([keep, sparse_rows], ignore_index=True)

    results = mlc.cross_validate_models(df, n_splits=5)
    for metrics in results.values():
        assert metrics["n_splits"] == 3


def test_train_and_evaluate_returns_confusion_matrix_and_report_for_both_models():
    df = _synthetic_training_df(rows_per_class=10)
    report, fitted_models = mlc.train_and_evaluate(df)

    assert set(report.keys()) == {"logistic_regression", mlc._secondary_model_name()}
    assert set(fitted_models.keys()) == set(report.keys())
    for name, metrics in report.items():
        assert 0.0 <= metrics["test_f1_macro"] <= 1.0
        cm = metrics["confusion_matrix"]
        assert set(cm.keys()) == {f"pred_{l}" for l in LABELS}
        assert "classification_report" in metrics


def test_feature_importance_table_covers_every_feature_column():
    df = _synthetic_training_df(rows_per_class=10)
    _, fitted_models = mlc.train_and_evaluate(df)
    for name, model in fitted_models.items():
        table = mlc.feature_importance_table(model, name)
        assert set(table["feature"]) == set(FEATURE_COLUMNS)
        assert len(table) == len(FEATURE_COLUMNS)
        # Sorted descending by importance.
        assert list(table["importance"]) == sorted(table["importance"], reverse=True)


def test_train_writes_a_low_row_count_warning_below_the_stability_floor(monkeypatch, tmp_path):
    df = _synthetic_training_df(rows_per_class=5)  # 15 rows, well under MIN_ROWS_FOR_TRAINING=40
    assert len(df) < mlc.MIN_ROWS_FOR_TRAINING

    csv_path = tmp_path / "training.csv"
    df.to_csv(csv_path, index=False)
    monkeypatch.setattr(mlc, "MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setattr(mlc, "METRICS_PATH", str(tmp_path / "metrics.json"))

    metrics = mlc.train(str(csv_path))

    assert metrics["n_rows"] == len(df)
    assert metrics["warning"] is not None
    assert "below" in metrics["warning"].lower()
    assert metrics["best_model"] in ("logistic_regression", mlc._secondary_model_name())
    assert {f["feature"] for f in metrics["feature_importances"]} == set(FEATURE_COLUMNS)
    assert os.path.exists(tmp_path / "model.joblib")
    assert os.path.exists(tmp_path / "metrics.json")


def test_train_omits_the_warning_at_or_above_the_stability_floor(monkeypatch, tmp_path):
    df = _synthetic_training_df(rows_per_class=14)  # 42 rows, >= MIN_ROWS_FOR_TRAINING=40
    assert len(df) >= mlc.MIN_ROWS_FOR_TRAINING

    csv_path = tmp_path / "training.csv"
    df.to_csv(csv_path, index=False)
    monkeypatch.setattr(mlc, "MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setattr(mlc, "METRICS_PATH", str(tmp_path / "metrics.json"))

    metrics = mlc.train(str(csv_path))
    assert metrics["warning"] is None


def test_predict_verdict_round_trips_against_a_freshly_trained_tmp_model(monkeypatch, tmp_path):
    """End-to-end: train a model to a tmp path (never touching the real
    committed artifact), then confirm predict_verdict can load and use
    it -- verifying train_final_model_on_all_data's saved bundle shape
    (model/model_name/feature_columns) matches what predict_verdict
    expects to read back."""
    df = _synthetic_training_df(rows_per_class=10)
    csv_path = tmp_path / "training.csv"
    df.to_csv(csv_path, index=False)
    monkeypatch.setattr(mlc, "MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setattr(mlc, "METRICS_PATH", str(tmp_path / "metrics.json"))

    mlc.train(str(csv_path))

    # A feature row planted squarely in class 2's (OVERVALUED) cluster
    # center should be classified as that class by a model trained on
    # cleanly-separated synthetic clusters.
    row = {c: 2 * 3.0 for c in FEATURE_COLUMNS}
    result = predict_verdict(row)
    assert result is not None
    assert result["verdict"] == LABELS[2]
