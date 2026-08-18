"""
Tests for app/valuation/ml_valuation_classifier.py -- exercises the
real train/evaluate pipeline against a small synthetic labeled
dataset (not scripts/ml_training_set.csv itself, which is real,
growing production data and shouldn't be a test dependency), so these
stay fast and deterministic regardless of what's actually been
collected in production.
"""

import numpy as np
import pandas as pd
import pytest

from app.valuation import ml_valuation_classifier as classifier
from app.valuation.ml_features import FEATURE_COLUMNS


def make_synthetic_training_df(n_per_label: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    labels = ["UNDERVALUED", "FAIRLY VALUED", "OVERVALUED"]
    rows = []
    for label in labels:
        # Give each label's features a distinct-ish mean so the models
        # have real signal to find -- an all-noise dataset would make
        # every downstream metric (F1, SHAP, ablations) meaningless.
        offset = {"UNDERVALUED": 1.0, "FAIRLY VALUED": 0.0, "OVERVALUED": -1.0}[label]
        for _ in range(n_per_label):
            row = {col: float(rng.normal(loc=offset, scale=0.5)) for col in FEATURE_COLUMNS}
            row["realized_label"] = label
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_df():
    return make_synthetic_training_df()


def test_load_training_data_drops_rows_missing_required_columns(tmp_path):
    df = make_synthetic_training_df(n_per_label=2)
    df.loc[0, FEATURE_COLUMNS[0]] = None
    path = tmp_path / "training.csv"
    df.to_csv(path, index=False)

    loaded = classifier.load_training_data(str(path))

    assert len(loaded) == len(df) - 1


def test_load_training_data_raises_on_missing_feature_column(tmp_path):
    df = make_synthetic_training_df(n_per_label=2).drop(columns=[FEATURE_COLUMNS[0]])
    path = tmp_path / "training.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match=FEATURE_COLUMNS[0]):
        classifier.load_training_data(str(path))


def test_cross_validate_models_reports_both_model_families(synthetic_df):
    results = classifier.cross_validate_models(synthetic_df)

    assert "logistic_regression" in results
    assert classifier._secondary_model_name() in results
    for metrics in results.values():
        assert 0.0 <= metrics["cv_accuracy_mean"] <= 1.0
        assert metrics["n_splits"] >= 2


def test_train_and_evaluate_returns_test_split_alongside_fitted_models(synthetic_df):
    report, fitted_models, X_test, y_test = classifier.train_and_evaluate(synthetic_df, test_size=0.3)

    assert set(fitted_models) == set(report)
    assert len(X_test) == len(y_test)
    # held out, not the full dataset -- confirms this is really a split
    assert len(X_test) < len(synthetic_df)
    for name, model in fitted_models.items():
        assert hasattr(model, "predict_proba")


def test_train_returns_artifacts_with_a_fitted_best_model(tmp_path, monkeypatch, synthetic_df):
    path = tmp_path / "training.csv"
    synthetic_df.to_csv(path, index=False)
    monkeypatch.setattr(classifier, "MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setattr(classifier, "METRICS_PATH", str(tmp_path / "metrics.json"))

    artifacts = classifier.train(str(path))

    assert artifacts.best_model_name in artifacts.metrics["held_out_test"]
    assert list(artifacts.X_test.columns) == FEATURE_COLUMNS
    assert len(artifacts.X_test) == len(artifacts.y_test)
    assert len(artifacts.df) == len(synthetic_df)
    preds = artifacts.best_model.predict(artifacts.X_test)
    assert len(preds) == len(artifacts.X_test)


def test_train_warns_below_the_minimum_row_floor(tmp_path, monkeypatch):
    small_df = make_synthetic_training_df(n_per_label=3)  # 9 rows, well under MIN_ROWS_FOR_TRAINING
    path = tmp_path / "training.csv"
    small_df.to_csv(path, index=False)
    monkeypatch.setattr(classifier, "MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setattr(classifier, "METRICS_PATH", str(tmp_path / "metrics.json"))

    artifacts = classifier.train(str(path))

    assert artifacts.metrics["warning"] is not None
    assert str(len(small_df)) in artifacts.metrics["warning"]
