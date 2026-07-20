"""
train_ml_classifier.py

Entry point: trains the ML valuation classifier on
scripts/ml_training_set.csv (produced by build_ml_training_set.py)
and prints the honest metrics -- cross-validated accuracy, held-out
per-class precision/recall/F1, not one blended number.

Usage:
    python scripts/build_ml_training_set.py    # generate labeled data first
    python scripts/train_ml_classifier.py       # train + report
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.valuation.ml_valuation_classifier import train, METRICS_PATH, MODEL_PATH

DATA_PATH = str(Path(__file__).resolve().parent / "ml_training_set.csv")


def main():
    metrics = train(DATA_PATH)

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
    print("\nFeature importances:")
    for row in metrics["feature_importances"]:
        print(f"  {row['feature']:<30} {row['importance']:.4f}")

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
