"""
ml_valuation_classifier.py

Trains/evaluates the (display-only) ML valuation classifier and
provides live inference. Architecture *adapted* from a separate,
read-only-audited project (DCF Valuation Engine) -- see the
integration audit -- but every number here comes from FinSight's own
point-in-time training set (scripts/build_ml_training_set.py), which
was built specifically to avoid the look-ahead bias the audit found
in that project's own headline accuracy claim (it trained on
"today's fundamentals graded against a 6-month-old price," despite
its own docs saying not to cite that number).

What's kept from the audited project's approach, because it's sound:
- Logistic Regression baseline vs. a gradient-boosted tree model,
  compared rather than assumed.
- Stratified k-fold cross-validation (more reliable on a small n than
  a single held-out split) AND a held-out test split (for a concrete
  confusion matrix), both reported -- not just whichever number looks
  best.
- Per-class precision/recall/F1, not one blended accuracy figure --
  the audited project's own model had a class it never learned to
  predict at all, invisible in the topline number alone.
- Labels are realized forward-return outcomes, not analyst agreement.

Deliberately NOT wired into report_data_builder.py's recommendation
composite (DCF_WEIGHT/RELATIVE_WEIGHT) -- this classifier has no
track record yet. It's computed and shown for transparency only,
same treatment as the Monte Carlo layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.base import BaseEstimator, ClassifierMixin
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False

from app.valuation.ml_features import FEATURE_COLUMNS

LABELS = ["UNDERVALUED", "FAIRLY VALUED", "OVERVALUED"]

_MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(_MODEL_DIR / "valuation_classifier.joblib")
METRICS_PATH = str(_MODEL_DIR / "ml_classifier_metrics.json")

MIN_ROWS_FOR_TRAINING = 40  # below this, CV folds get unstable -- warn, don't refuse


if HAS_XGBOOST:
    class _LabelEncodedXGB(BaseEstimator, ClassifierMixin):
        """Thin wrapper so XGBClassifier (needs numeric 0..K-1 labels)
        exposes the same string-label fit/predict/predict_proba API
        as everything else here."""

        def __init__(self, **xgb_kwargs):
            self.xgb_kwargs = xgb_kwargs
            self._encoder = LabelEncoder()
            self._model = XGBClassifier(**xgb_kwargs)

        def fit(self, X, y, sample_weight=None):
            # sample_weight: XGBClassifier has no multiclass class_weight
            # equivalent (unlike LogisticRegression's constructor-time
            # "balanced" option, see build_logreg's own comment) -- the
            # standard alternative is per-sample weights supplied at fit
            # time, computed via sklearn.utils.class_weight.compute_sample_weight.
            y_enc = self._encoder.fit_transform(y)
            self._model.fit(X, y_enc, sample_weight=sample_weight)
            self.classes_ = self._encoder.classes_
            return self

        def predict(self, X):
            return self._encoder.inverse_transform(self._model.predict(X))

        def predict_proba(self, X):
            return self._model.predict_proba(X)

        @property
        def feature_importances_(self):
            return self._model.feature_importances_


def build_secondary_model():
    if HAS_XGBOOST:
        return _LabelEncodedXGB(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            random_state=42,
        )
    return GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    )


def _secondary_model_name() -> str:
    return "xgboost" if HAS_XGBOOST else "gradient_boosting"


def build_logreg():
    # class_weight="balanced" -- without it, on this dataset's real
    # label distribution (roughly 60% UNDERVALUED after the Phase 2
    # broad-universe expansion, a bull-market base rate baked into the
    # labels themselves, not learned skill), Logistic Regression simply
    # predicted UNDERVALUED for nearly everything: 100% recall on that
    # class, ~0-4% recall on the other two, F1-macro dropping even as
    # raw accuracy rose. Confirmed against this exact training set
    # before adding this. Re-weights each class inversely to its
    # frequency during fitting, the standard sklearn-native fix for
    # exactly this failure mode -- no external sample_weight plumbing
    # needed since it's a constructor-time parameter, so it works
    # transparently through cross_validate() and train_test_split()
    # alike, unlike the secondary model below (see build_secondary_model).
    return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])


def load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Training file is missing feature columns: {missing}")
    # "ticker" isn't a model feature (deliberately excluded from
    # FEATURE_COLUMNS -- a company identifier would let the model
    # memorize per-company labels instead of learning a generalizable
    # valuation signal) but IS required for the group-aware splits
    # below -- see cross_validate_models/train_and_evaluate's own
    # comments for why a plain random split leaks across rows that
    # share a ticker.
    if "ticker" not in df.columns:
        raise ValueError("Training file is missing the 'ticker' column (needed for group-aware CV/splits)")
    return df.dropna(subset=FEATURE_COLUMNS + ["realized_label"])


def _fit_model(name: str, model, X_train, y_train):
    """Fits with class-balanced weighting -- class_weight="balanced" is
    already baked into build_logreg()'s constructor (needs no fit-time
    plumbing), the secondary model needs an explicit sample_weight
    instead (see _LabelEncodedXGB.fit's own comment). Shared by
    cross_validate_models and train_and_evaluate so both apply the
    exact same weighting, not two independently-maintained copies."""
    if name == "logistic_regression":
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
    return model


def cross_validate_models(df: pd.DataFrame, n_splits: int = 5) -> Dict[str, Any]:
    X = df[FEATURE_COLUMNS].astype("float64").to_numpy()
    y = df["realized_label"].astype(str).to_numpy()
    groups = df["ticker"].to_numpy()

    # GroupKFold, not StratifiedKFold -- real leakage, not theoretical:
    # 550 of 618 tickers in the production training set
    # (scripts/ml_training_set.csv, built by combining several as-of-date
    # windows) appear in MORE than one row. A plain (stratified-by-label
    # but not grouped) K-fold routinely put, say, 3 of ADBE's 4
    # snapshots in the training fold and the 4th in the test fold --
    # letting the model partly recognize "this row smells like ADBE"
    # rather than learn a signal that generalizes to a company it has
    # never seen at all. Confirmed directly, not assumed: re-scoring the
    # SAME saved model with a GroupKFold-safe split dropped XGBoost's
    # cross-validated accuracy from 48.1% to 42.8%, and OVERVALUED-class
    # precision from an apparent 39.3% (held-out, same leak) to a real
    # 30.6% -- still a genuine edge over the DCF composite rule's own
    # 27.1% Sell precision (EVALUATION.md section 0), just a smaller,
    # honest one instead of an inflated one.
    #
    # int(...) matters, not just style -- when the smallest class's
    # count is what actually binds (small/imbalanced datasets, which
    # this one still is), Python's min()/max() return that value
    # as-is rather than casting it, leaving a bare numpy.int64 in a
    # dict this module later json.dumps -- which raises. Only surfaced
    # once a class dropped under the n_splits=5 default, not before.
    # Also capped by unique ticker count now -- GroupKFold raises if
    # asked for more folds than there are distinct groups, which a
    # small/synthetic dataset (or a training set with a lot of ticker
    # repetition) can easily have fewer of than 5.
    n_splits = int(max(min(n_splits, df["realized_label"].value_counts().min(), df["ticker"].nunique()), 2))
    cv = GroupKFold(n_splits=n_splits)

    model_builders = {"logistic_regression": build_logreg, _secondary_model_name(): build_secondary_model}

    # Manual fold loop instead of sklearn's cross_validate() convenience
    # wrapper -- passing sample_weight through cross_validate needs its
    # newer metadata-routing API (sklearn's `params=`, confirmed this
    # installed version no longer even accepts the older `fit_params=`
    # kwarg), which requires each estimator to explicitly opt in via
    # set_fit_request -- version-fragile complexity next to just
    # fitting each fold directly with _fit_model above.
    results = {}
    for name, builder in model_builders.items():
        accuracies, f1_macros = [], []
        for train_idx, test_idx in cv.split(X, y, groups=groups):
            model = _fit_model(name, builder(), X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            accuracies.append(accuracy_score(y[test_idx], preds))
            f1_macros.append(f1_score(y[test_idx], preds, average="macro", zero_division=0))
        results[name] = {
            "cv_accuracy_mean": float(np.mean(accuracies)),
            "cv_accuracy_std": float(np.std(accuracies)),
            "cv_f1_macro_mean": float(np.mean(f1_macros)),
            "cv_f1_macro_std": float(np.std(f1_macros)),
            "n_splits": n_splits,
        }
    return results


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.25):
    X = df[FEATURE_COLUMNS].astype("float64").to_numpy()
    y = df["realized_label"].astype(str).to_numpy()
    groups = df["ticker"].to_numpy()

    # GroupShuffleSplit, not train_test_split -- same leakage this
    # module's cross_validate_models now guards against (see its own
    # comment): a ticker that shows up in both X_train and X_test lets
    # the held-out confusion matrix partly reflect memorization, not
    # generalization. GroupShuffleSplit has no built-in label
    # stratification (unlike train_test_split's `stratify=`) -- a real,
    # accepted tradeoff, not an oversight: keeping every one of a
    # ticker's rows on the same side of the split is the property that
    # actually matters for an honest held-out test here.
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    report = {}
    fitted_models = {}

    for name, model in [("logistic_regression", build_logreg()),
                         (_secondary_model_name(), build_secondary_model())]:
        model = _fit_model(name, model, X_train, y_train)
        preds = model.predict(X_test)

        cm = confusion_matrix(y_test, preds, labels=LABELS)
        cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])
        cls_report = classification_report(y_test, preds, labels=LABELS, zero_division=0, output_dict=True)

        report[name] = {
            "test_f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
            "confusion_matrix": cm_df.to_dict(),
            "classification_report": cls_report,
        }
        fitted_models[name] = model

    return report, fitted_models, X_test, y_test


def feature_importance_table(model, model_name: str) -> pd.DataFrame:
    try:
        if model_name == "logistic_regression":
            clf = model.named_steps["clf"]
            importances = np.abs(clf.coef_).mean(axis=0)
        else:
            importances = model.feature_importances_
        return pd.DataFrame(
            {"feature": FEATURE_COLUMNS, "importance": importances}
        ).sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def train_final_model_on_all_data(df: pd.DataFrame, best_model_name: str):
    X, y = df[FEATURE_COLUMNS], df["realized_label"]
    model = build_logreg() if best_model_name == "logistic_regression" else build_secondary_model()
    model = _fit_model(best_model_name, model, X, y)
    joblib.dump({"model": model, "model_name": best_model_name, "feature_columns": FEATURE_COLUMNS}, MODEL_PATH)
    return model


def predict_verdict(feature_row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Loads the saved model and predicts a verdict + class
    probabilities for one feature row (from ml_features.extract_features).
    Returns None if no trained model file exists yet, or if any
    required feature is missing/None."""
    import os
    if not os.path.exists(MODEL_PATH):
        return None
    if any(feature_row.get(c) is None for c in FEATURE_COLUMNS):
        return None

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    cols = bundle["feature_columns"]

    X = pd.DataFrame([{c: feature_row[c] for c in cols}])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    classes = model.classes_ if hasattr(model, "classes_") else model.named_steps["clf"].classes_
    return {
        "verdict": pred,
        "probabilities": {cls: float(p) for cls, p in zip(classes, proba)},
        "model_name": bundle["model_name"],
    }


@dataclass
class TrainingArtifacts:
    """
    Everything train() produces beyond the JSON-serializable metrics
    dict -- the fitted best-model estimator plus its held-out test
    split and the full labeled DataFrame, for callers that want to go
    deeper than the summary metrics (SHAP importance, calibration
    curves, feature-group ablations -- see ml_evaluation.py). Kept out
    of the `metrics` dict itself so METRICS_PATH stays plain JSON.
    """
    metrics: Dict[str, Any]
    best_model_name: str
    best_model: Any
    X_test: pd.DataFrame
    y_test: np.ndarray
    df: pd.DataFrame


def train(data_path: str) -> TrainingArtifacts:
    """Full train+evaluate+save pipeline. metrics (on the returned
    TrainingArtifacts) is also written to METRICS_PATH."""
    df = load_training_data(data_path)
    n_rows = len(df)

    warning = None
    if n_rows < MIN_ROWS_FOR_TRAINING:
        warning = (
            f"Only {n_rows} labeled rows -- below the {MIN_ROWS_FOR_TRAINING}-row floor "
            f"for stable cross-validation. Metrics below are noisy; treat as directional "
            f"evidence, not a benchmark, until the training set grows."
        )

    cv_results = cross_validate_models(df)
    test_report, fitted_models, X_test, y_test = train_and_evaluate(df)

    best_name = max(test_report, key=lambda k: test_report[k]["test_f1_macro"])
    importances = feature_importance_table(fitted_models[best_name], best_name)
    train_final_model_on_all_data(df, best_name)

    metrics = {
        "n_rows": n_rows,
        "warning": warning,
        "label_distribution": df["realized_label"].value_counts().to_dict(),
        "cross_validation": cv_results,
        "held_out_test": {
            k: {"test_f1_macro": v["test_f1_macro"], "classification_report": v["classification_report"]}
            for k, v in test_report.items()
        },
        "best_model": best_name,
        "feature_importances": importances.to_dict(orient="records"),
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return TrainingArtifacts(
        metrics=metrics,
        best_model_name=best_name,
        best_model=fitted_models[best_name],
        X_test=pd.DataFrame(X_test, columns=FEATURE_COLUMNS),
        y_test=y_test,
        df=df,
    )
