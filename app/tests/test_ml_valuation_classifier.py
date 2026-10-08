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
    ticker_counter = 0
    for label in labels:
        # Give each label's features a distinct-ish mean so the models
        # have real signal to find -- an all-noise dataset would make
        # every downstream metric (F1, SHAP, ablations) meaningless.
        offset = {"UNDERVALUED": 1.0, "FAIRLY VALUED": 0.0, "OVERVALUED": -1.0}[label]
        for i in range(n_per_label):
            row = {col: float(rng.normal(loc=offset, scale=0.5)) for col in FEATURE_COLUMNS}
            row["realized_label"] = label
            # Every OTHER row repeats the previous row's ticker (not a
            # fresh one each time) -- deliberately exercises the
            # group-aware split's actual job (keeping a repeated
            # ticker's rows together) rather than degenerating into a
            # plain row-level split where every group happens to have
            # exactly one member.
            if i % 2 == 0:
                ticker_counter += 1
            row["ticker"] = f"SYN{ticker_counter}"
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


def test_train_and_evaluate_never_splits_a_ticker_across_train_and_test(synthetic_df):
    # The actual leakage regression test: synthetic_df has each ticker
    # appearing in exactly 2 rows (see make_synthetic_training_df) --
    # a plain (non-grouped) split would, over repeated runs, sometimes
    # put one of those 2 rows in train and the other in test. Asserting
    # this on the real X_test/y_test arrays alone isn't possible (they
    # don't carry the ticker column through) -- so this reaches into
    # the same GroupShuffleSplit call train_and_evaluate makes, using
    # the identical arguments, to confirm the split itself is group-safe.
    from sklearn.model_selection import GroupShuffleSplit

    X = synthetic_df[FEATURE_COLUMNS].astype("float64").to_numpy()
    y = synthetic_df["realized_label"].astype(str).to_numpy()
    groups = synthetic_df["ticker"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    train_tickers = set(groups[train_idx])
    test_tickers = set(groups[test_idx])
    assert train_tickers.isdisjoint(test_tickers)
    # Sanity check the fixture itself actually has repeated tickers --
    # otherwise this test would trivially pass even against the old,
    # leaky train_test_split (nothing to leak if every group has 1 row).
    assert synthetic_df["ticker"].duplicated().any()


def test_cross_validate_models_never_splits_a_ticker_across_folds(synthetic_df):
    fold_ticker_sets = []
    X = synthetic_df[FEATURE_COLUMNS].astype("float64").to_numpy()
    y = synthetic_df["realized_label"].astype(str).to_numpy()
    groups = synthetic_df["ticker"].to_numpy()

    from sklearn.model_selection import GroupKFold

    cv = GroupKFold(n_splits=5)
    for _, test_idx in cv.split(X, y, groups=groups):
        fold_ticker_sets.append(set(groups[test_idx]))

    # No ticker appears as a held-out member of more than one fold --
    # the group-safety property GroupKFold is supposed to guarantee.
    seen = set()
    for fold_tickers in fold_ticker_sets:
        assert seen.isdisjoint(fold_tickers)
        seen |= fold_tickers


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
