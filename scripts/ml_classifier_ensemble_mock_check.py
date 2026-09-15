"""
ml_classifier_ensemble_mock_check.py

MOCK feasibility check, not a production build: does gating Sell calls
with the (now leakage-fixed, see ml_valuation_classifier.py's own
comment) ML classifier's OVERVALUED prediction move the composite
rule's pooled accuracy enough to justify actually building the real
integration (a merged canonical-backtest + ML-feature dataset, wired
into report_data_builder.py, its own test coverage)?

Self-contained within scripts/ml_training_set.csv alone -- no merge
with the separate canonical-backtest artifacts needed for this check,
because ml_training_set.csv already carries everything both signals
need: the raw features the ML classifier trains on, AND
dcf_over_price/relative_vs_history_pct, which are exactly what
report_data_builder._dcf_score/_relative_score/_composite_score need
to reproduce the SAME composite rating the live app computes. Reusing
those real functions (not reimplementing the formula) is the same
"don't invent a second copy that can drift" discipline
tune_momentum_weight.py already established.

Fairness: the ML classifier's prediction for each row comes from
out-of-fold GroupKFold cross-validation (grouped by ticker, same
leakage-safety this session just fixed in ml_valuation_classifier.py)
-- every row is scored by a model that never saw that row's ticker
during training, not by the model that was fit on the whole dataset.

realized_label (UNDERVALUED/FAIRLY VALUED/OVERVALUED) IS the ground
truth for what rating would have been objectively correct -- built
from the exact same +-5% realized-return rule as Buy/Hold/Sell
(scripts/build_ml_training_set.py's own realized_label() function) --
so "correct" here means composite_rating (or the ensemble's rating)
equals the Buy/Hold/Sell-equivalent of realized_label, mapped
UNDERVALUED<->Buy, FAIRLY VALUED<->Hold, OVERVALUED<->Sell.

Run: python scripts/ml_classifier_ensemble_mock_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from app.reporting.report_data_builder import _composite_score, _dcf_score, _rating_from_score, _relative_score
from app.valuation.ml_features import FEATURE_COLUMNS
from app.valuation.ml_valuation_classifier import _fit_model, _secondary_model_name, build_secondary_model

DATA_PATH = str(Path(__file__).resolve().parent / "ml_training_set.csv")

_LABEL_TO_RATING = {"UNDERVALUED": "Buy", "FAIRLY VALUED": "Hold", "OVERVALUED": "Sell"}


def composite_rating(row) -> str:
    upside_percent = (row["dcf_over_price"] - 1.0) * 100.0
    dcf_score = _dcf_score(upside_percent)
    vs_history_pct = row["relative_vs_history_pct"]
    relative_score = _relative_score({"vs_history_pct": vs_history_pct}) if pd.notna(vs_history_pct) else None
    composite = _composite_score(dcf_score, relative_score)
    return _rating_from_score(composite)


def out_of_fold_ml_predictions(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLUMNS].astype("float64").to_numpy()
    y = df["realized_label"].astype(str).to_numpy()
    groups = df["ticker"].to_numpy()

    preds = np.empty(len(df), dtype=object)
    cv = GroupKFold(n_splits=5)
    model_name = _secondary_model_name()
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        model = _fit_model(model_name, build_secondary_model(), X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
    return preds


def accuracy(ratings, realized_labels) -> tuple:
    correct = sum(1 for r, lbl in zip(ratings, realized_labels) if r == _LABEL_TO_RATING[lbl])
    return 100 * correct / len(ratings), len(ratings)


def precision_of_rating(ratings, realized_labels, target_rating: str) -> tuple:
    idx = [i for i, r in enumerate(ratings) if r == target_rating]
    if not idx:
        return None, 0
    correct = sum(1 for i in idx if _LABEL_TO_RATING[realized_labels[i]] == target_rating)
    return 100 * correct / len(idx), len(idx)


def main():
    print(f"Loading {DATA_PATH}...", file=sys.stderr)
    df = pd.read_csv(DATA_PATH).dropna(subset=FEATURE_COLUMNS + ["realized_label", "dcf_over_price"])
    print(f"Usable rows: {len(df)}  unique tickers: {df['ticker'].nunique()}", file=sys.stderr)

    df = df.reset_index(drop=True)
    df["composite_rating"] = df.apply(composite_rating, axis=1)

    print("\nGenerating out-of-fold ML classifier predictions (GroupKFold by ticker, leak-safe)...", file=sys.stderr)
    df["ml_prediction"] = out_of_fold_ml_predictions(df)

    realized = df["realized_label"].to_numpy()

    print("\n--- Baseline: composite rule alone ---", file=sys.stderr)
    base_acc, base_n = accuracy(df["composite_rating"], realized)
    base_sell_prec, base_sell_n = precision_of_rating(df["composite_rating"], realized, "Sell")
    print(f"  Pooled accuracy: {base_acc:.1f}% (N={base_n})", file=sys.stderr)
    print(f"  Sell precision: {base_sell_prec:.1f}% (n={base_sell_n})", file=sys.stderr)

    # Ensemble A: composite Sell only survives if the ML classifier's
    # OUT-OF-FOLD prediction agrees (OVERVALUED); otherwise downgraded
    # to Hold. Same "suppress, don't invent a new rating" shape as the
    # Financials/Energy check, but gated by a genuinely independent
    # second model instead of a sector lookup table.
    ensemble_a = [
        "Hold" if (row.composite_rating == "Sell" and row.ml_prediction != "OVERVALUED") else row.composite_rating
        for row in df.itertuples()
    ]
    acc_a, n_a = accuracy(ensemble_a, realized)
    sell_prec_a, sell_n_a = precision_of_rating(ensemble_a, realized, "Sell")
    print("\n--- Ensemble A: Sell requires composite AND ml_prediction=OVERVALUED agreement ---", file=sys.stderr)
    print(f"  Pooled accuracy: {acc_a:.1f}% (N={n_a})  delta={acc_a - base_acc:+.2f}pt", file=sys.stderr)
    print(f"  Sell precision: {sell_prec_a:.1f}% (n={sell_n_a})", file=sys.stderr)

    # Ensemble B: symmetric version -- also require agreement for Buy
    # (UNDERVALUED), not just Sell, to check whether gating helps or
    # hurts the model's already-strongest signal.
    ensemble_b = []
    for row in df.itertuples():
        if row.composite_rating == "Sell" and row.ml_prediction != "OVERVALUED":
            ensemble_b.append("Hold")
        elif row.composite_rating == "Buy" and row.ml_prediction != "UNDERVALUED":
            ensemble_b.append("Hold")
        else:
            ensemble_b.append(row.composite_rating)
    acc_b, n_b = accuracy(ensemble_b, realized)
    print("\n--- Ensemble B: same gating applied symmetrically to Buy too ---", file=sys.stderr)
    print(f"  Pooled accuracy: {acc_b:.1f}% (N={n_b})  delta={acc_b - base_acc:+.2f}pt", file=sys.stderr)

    # Ensemble C: ML classifier used as the SOLE Sell signal (not a
    # gate on the composite) -- i.e. would a stock ever get called
    # Sell if only the ML model's own OVERVALUED prediction decided it,
    # regardless of what the DCF composite says.
    ensemble_c = [
        "Sell" if row.ml_prediction == "OVERVALUED" else ("Buy" if row.ml_prediction == "UNDERVALUED" else "Hold")
        for row in df.itertuples()
    ]
    acc_c, n_c = accuracy(ensemble_c, realized)
    sell_prec_c, sell_n_c = precision_of_rating(ensemble_c, realized, "Sell")
    print("\n--- Ensemble C: ML classifier's own prediction used directly as the rating (no DCF at all) ---", file=sys.stderr)
    print(f"  Pooled accuracy: {acc_c:.1f}% (N={n_c})  delta={acc_c - base_acc:+.2f}pt", file=sys.stderr)
    print(f"  Sell precision: {sell_prec_c:.1f}% (n={sell_n_c})", file=sys.stderr)

    print(
        "\nVerdict threshold for this mock check: +2.0pt or more of pooled-accuracy gain "
        "(same bar the Financials/Energy sector check used) before it's worth building the "
        "real merged-dataset + production-wiring version.",
        file=sys.stderr,
    )
    best_delta = max(acc_a - base_acc, acc_b - base_acc, acc_c - base_acc)
    verdict = "WORTH BUILDING" if best_delta >= 2.0 else "NOT WORTH THE FULL BUILD"
    print(f"Best delta across all 3 ensemble variants: {best_delta:+.2f}pt -> {verdict}", file=sys.stderr)


if __name__ == "__main__":
    main()
