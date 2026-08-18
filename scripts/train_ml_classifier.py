"""
train_ml_classifier.py

Entry point: trains the ML valuation classifier on
scripts/ml_training_set.csv (produced by build_ml_training_set.py),
prints the honest metrics -- cross-validated accuracy, held-out
per-class precision/recall/F1, not one blended number -- and logs the
full run (params, metrics, confusion matrix, SHAP importance,
calibration curves, feature-group ablations, the fitted model itself)
to a local MLflow tracking store under mlruns/ so every training run
is reproducible and comparable to the ones before it, not just the
latest joblib file on disk.

Usage:
    python scripts/build_ml_training_set.py    # generate labeled data first
    python scripts/train_ml_classifier.py       # train + report + log to MLflow
    mlflow ui                                   # browse past runs (http://127.0.0.1:5000)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlflow

from app.valuation.ml_evaluation import calibration_data, run_feature_ablations, shap_importance_table
from app.valuation.ml_valuation_classifier import METRICS_PATH, MODEL_PATH, train

DATA_PATH = str(Path(__file__).resolve().parent / "ml_training_set.csv")
MLRUNS_DIR = Path(__file__).resolve().parent.parent / "mlruns"


def _log_to_mlflow(artifacts) -> None:
    # MLflow 3.x refuses the plain filesystem store by default (now
    # "maintenance mode only") -- a local SQLite file is the modern
    # equivalent: still fully local/free/no server, but gets the
    # current MLflow feature set instead of the deprecated backend.
    MLRUNS_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}")
    mlflow.set_experiment("finsight-valuation-classifier")

    metrics = artifacts.metrics
    with mlflow.start_run():
        mlflow.log_params({
            "n_rows": metrics["n_rows"],
            "best_model": artifacts.best_model_name,
            "n_features": len(artifacts.X_test.columns),
        })

        for model_name, cv in metrics["cross_validation"].items():
            mlflow.log_metrics({
                f"cv_accuracy_mean__{model_name}": cv["cv_accuracy_mean"],
                f"cv_f1_macro_mean__{model_name}": cv["cv_f1_macro_mean"],
            })
        for model_name, test_result in metrics["held_out_test"].items():
            mlflow.log_metric(f"test_f1_macro__{model_name}", test_result["test_f1_macro"])

        best_report = metrics["held_out_test"][artifacts.best_model_name]["classification_report"]
        for label in ("UNDERVALUED", "FAIRLY VALUED", "OVERVALUED"):
            per_class = best_report.get(label)
            if not per_class:
                continue
            mlflow.log_metrics({
                f"best_model_precision__{label}": per_class["precision"],
                f"best_model_recall__{label}": per_class["recall"],
                f"best_model_f1__{label}": per_class["f1-score"],
            })

        shap_table = shap_importance_table(artifacts.best_model, artifacts.X_test)
        calibration = calibration_data(artifacts.best_model, artifacts.X_test, artifacts.y_test)
        ablations = run_feature_ablations(artifacts.df, artifacts.best_model_name)

        mlflow.log_dict(metrics, "metrics.json")
        mlflow.log_dict(shap_table.to_dict(orient="records"), "shap_importance.json")
        mlflow.log_dict(calibration, "calibration.json")
        mlflow.log_dict(ablations, "ablations.json")

        # mlflow's skops-based serializer refuses to trust custom
        # classes by default (a real security feature against loading
        # arbitrary code from someone else's model file) -- our own
        # xgboost wrapper and xgboost's own classes are the only ones
        # that trip it here, and this is our own model we just trained
        # in this same process, so trusting them is correct, not a
        # workaround.
        mlflow.sklearn.log_model(
            artifacts.best_model,
            name="model",
            skops_trusted_types=[
                "app.valuation.ml_valuation_classifier._LabelEncodedXGB",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )

    return shap_table, calibration, ablations


def main():
    artifacts = train(DATA_PATH)
    metrics = artifacts.metrics

    print(f"Trained on {metrics['n_rows']} rows")
    if metrics.get("warning"):
        print(f"[warning] {metrics['warning']}")
    print(f"Label distribution: {metrics['label_distribution']}")

    print("\n=== Cross-validated comparison ===")
    print(json.dumps(metrics["cross_validation"], indent=2))

    print("\n=== Held-out test set ===")
    for name, result in metrics["held_out_test"].items():
        print(f"\n--- {name} ---")
        cr = result["classification_report"]
        print(f"  accuracy: {cr['accuracy']:.1%}   macro F1: {result['test_f1_macro']:.3f}")
        for label in ("UNDERVALUED", "FAIRLY VALUED", "OVERVALUED"):
            c = cr.get(label, {})
            print(f"  {label:<15} precision={c.get('precision', 0):.2f}  "
                  f"recall={c.get('recall', 0):.2f}  f1={c.get('f1-score', 0):.2f}  "
                  f"support={c.get('support', 0):.0f}")

    print(f"\nBest model: {metrics['best_model']}")
    print("\nFeature importances (coefficient/gain):")
    for row in metrics["feature_importances"]:
        print(f"  {row['feature']:<30} {row['importance']:.4f}")

    shap_table, calibration, ablations = _log_to_mlflow(artifacts)

    print("\nFeature importances (SHAP, model-agnostic):")
    for _, row in shap_table.iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")

    print("\nCalibration (per-class, predicted vs. actual):")
    for label, points in calibration.items():
        print(f"  {label}: predicted={[f'{p:.2f}' for p in points['mean_predicted']]}  "
              f"actual={[f'{p:.2f}' for p in points['fraction_positive']]}")

    print(f"\nFeature-group ablations (baseline macro F1 = {ablations['baseline_macro_f1']:.3f}):")
    for group, result in ablations["ablations"].items():
        sign = "+" if result["macro_f1_delta"] >= 0 else ""
        print(f"  without {group:<15} ({', '.join(result['removed_features'])}): "
              f"{result['macro_f1_without_group']:.3f}  ({sign}{result['macro_f1_delta']:.3f})")

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Logged MLflow run -> {MLRUNS_DIR} (run `mlflow ui` to browse)")


if __name__ == "__main__":
    main()
