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
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

        def fit(self, X, y):
            y_enc = self._encoder.fit_transform(y)
            self._model.fit(X, y_enc)
            self.classes_ = self._encoder.classes_
            return self

        def predict(self, X):
            return self._encoder.inverse_transform(self._model.predict(X))

        def predict_proba(self, X):
            return self._model.predict_proba(X)

        @property
        def feature_importances_(self):
            return self._model.feature_importances_


def _build_secondary_model():
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


def _build_logreg():
    return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])


def load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Training file is missing feature columns: {missing}")
    return df.dropna(subset=FEATURE_COLUMNS + ["realized_label"])


def cross_validate_models(df: pd.DataFrame, n_splits: int = 5) -> Dict[str, Any]:
    X = df[FEATURE_COLUMNS].astype("float64").to_numpy()
    y = df["realized_label"].astype(str).to_numpy()

    n_splits = max(min(n_splits, df["realized_label"].value_counts().min()), 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = {"logistic_regression": _build_logreg(), _secondary_model_name(): _build_secondary_model()}

    results = {}
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1_macro"], return_train_score=False)
        results[name] = {
            "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "cv_accuracy_std": float(np.std(scores["test_accuracy"])),
            "cv_f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
            "cv_f1_macro_std": float(np.std(scores["test_f1_macro"])),
            "n_splits": n_splits,
        }
    return results


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.25):
    X = df[FEATURE_COLUMNS].astype("float64").to_numpy()
    y = df["realized_label"].astype(str).to_numpy()

    class_counts = np.unique(y, return_counts=True)[1]
    stratify = y if class_counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    report = {}
    fitted_models = {}

    for name, model in [("logistic_regression", _build_logreg()),
                         (_secondary_model_name(), _build_secondary_model())]:
        model.fit(X_train, y_train)
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

    return report, fitted_models


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
    model = _build_logreg() if best_model_name == "logistic_regression" else _build_secondary_model()
    model.fit(X, y)
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


def train(data_path: str) -> Dict[str, Any]:
    """Full train+evaluate+save pipeline. Returns the metrics dict
    that also gets written to METRICS_PATH."""
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
    test_report, fitted_models = train_and_evaluate(df)

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

    return metrics
