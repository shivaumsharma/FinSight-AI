"""
tune_ml_weight.py

Tests whether folding the ML valuation classifier into the
recommendation composite (as a third weighted term alongside DCF and
relative valuation) actually moves accuracy -- not assumed, measured,
same discipline as tune_recommendation_config.py's existing
DCF_WEIGHT/RELATIVE_WEIGHT grid search.

RESULT (see EVALUATION.md section 4 for the full writeup): it doesn't,
once evaluated honestly. Pooled accuracy across all 4 windows barely
moves (44.1% at ML_WEIGHT=0 -> 44.8% at the best-scoring weight, 0.3),
and every tested weight from 0.0 to 0.5 lands within 1 point of every
other -- a flat plateau, not a real effect in either direction. The
classifier is NOT wired into report_data_builder.py's composite as a
result of this test; DCF_WEIGHT/RELATIVE_WEIGHT are unchanged. Kept as
a script, not deleted, so this negative result is reproducible and the
question doesn't need re-litigating from scratch once more Phase 2
data exists.

Reuses the Phase 2 training-set sweep's already-computed data
(scripts/ml_training_set_ticker_universe_asof{12,18,24,30}mo.csv) --
each row already has the raw features AND realized_return_pct from a
genuine point-in-time backtest run, so this needs no new network calls
or pipeline re-runs. dcf_score/relative_score are reconstructed from
those raw features using report_data_builder.py's OWN _dcf_score/
_relative_score functions (not reimplemented), so this is testing the
exact same scoring the live app uses, just replayed offline.

ml_score definition: (P(UNDERVALUED) - P(OVERVALUED)) * 100 from the
classifier's predict_proba -- naturally bounded to [-100, 100] by
construction (probabilities sum to <=1), unlike raw upside_percent/
vs_history_pct, which is why THOSE needed percentile-rank
normalization (see report_data_builder.py's own "Percentile-based
normalization" comment) and this doesn't.

Leave-one-window-out, not the single already-saved model: the saved
valuation_classifier.joblib was trained on ALL 4 windows combined (see
ml_valuation_classifier.train_final_model_on_all_data) -- using it to
score these same 4 windows would silently evaluate the model on rows
it already memorized during its own training, inflating every
ML_WEIGHT's apparent benefit (confirmed while building this: an
earlier version of this script did exactly that and showed a
suspiciously large, monotonically-increasing benefit all the way to
ML_WEIGHT=1.0 -- a classic in-sample-leakage signature, the same class
of mistake as Phase 2's class-imbalance trap, just a different
flavor). Each window is instead scored using a classifier retrained
on the OTHER 3 windows only, genuinely out-of-sample for the window
being evaluated.

Run: python scripts/tune_ml_weight.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.analysis.baseline_scoring import score_rating
from app.reporting.report_data_builder import _dcf_score, _relative_score, _rating_from_score
from app.valuation.ml_features import FEATURE_COLUMNS
from app.valuation.ml_valuation_classifier import _fit_model, _secondary_model_name, build_secondary_model

SCRIPT_DIR = Path(__file__).resolve().parent
WINDOW_FILES = [
    "ml_training_set_ticker_universe_asof12mo.csv",
    "ml_training_set_ticker_universe_asof18mo.csv",
    "ml_training_set_ticker_universe_asof24mo.csv",
    "ml_training_set_ticker_universe_asof30mo.csv",
]

# Current production DCF/relative ratio (report_data_builder.py's
# DCF_WEIGHT=0.8/RELATIVE_WEIGHT=0.2) -- preserved between the two when
# ML_WEIGHT eats into the total, so this tests "does giving ML a slice
# help," not a simultaneous re-tune of the DCF/relative ratio too
# (already validated separately, see that module's own tuning comment).
PROD_DCF_RATIO = 0.8
PROD_RELATIVE_RATIO = 0.2

ML_WEIGHTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def load_window(filename: str) -> pd.DataFrame:
    df = pd.read_csv(SCRIPT_DIR / filename)
    df["upside_percent"] = (df["dcf_over_price"] - 1.0) * 100.0
    df["_window"] = filename
    return df


def _model_classes(model):
    return list(model.classes_) if hasattr(model, "classes_") else list(model.named_steps["clf"].classes_)


def compute_scores(df: pd.DataFrame, model) -> pd.DataFrame:
    df = df.copy()
    df["dcf_score"] = df["upside_percent"].apply(_dcf_score)
    df["relative_score"] = df["relative_vs_history_pct"].apply(
        lambda v: _relative_score({"vs_history_pct": v}) if pd.notna(v) else None
    )

    X = df[FEATURE_COLUMNS].astype("float64").to_numpy()
    proba = model.predict_proba(X)
    classes = _model_classes(model)
    under_idx = classes.index("UNDERVALUED")
    over_idx = classes.index("OVERVALUED")
    df["ml_score"] = (proba[:, under_idx] - proba[:, over_idx]) * 100.0
    return df


def composite_with_ml(row, ml_weight: float) -> float:
    dcf_weight = (1.0 - ml_weight) * PROD_DCF_RATIO
    relative_weight = (1.0 - ml_weight) * PROD_RELATIVE_RATIO
    relative_score = row["relative_score"]
    if pd.isna(relative_score):
        # Same renormalization spirit as report_data_builder.py's
        # _composite_score when relative valuation is unavailable --
        # redistribute its share proportionally between DCF and ML
        # rather than silently dropping it.
        total = dcf_weight + ml_weight
        if total == 0:
            return row["dcf_score"]
        return (dcf_weight / total) * row["dcf_score"] + (ml_weight / total) * row["ml_score"]
    return dcf_weight * row["dcf_score"] + relative_weight * relative_score + ml_weight * row["ml_score"]


def evaluate(df: pd.DataFrame, ml_weight: float):
    correct = 0
    scored = 0
    for _, row in df.iterrows():
        composite = composite_with_ml(row, ml_weight)
        rating = _rating_from_score(composite)
        result = score_rating(rating, row["realized_return_pct"])
        if result is None:
            continue
        scored += 1
        if result:
            correct += 1
    if scored == 0:
        return None
    return correct, scored, 100 * correct / scored


def main():
    model_name = _secondary_model_name()
    print(f"Model (retrained per held-out window): {model_name}   Features: {FEATURE_COLUMNS}\n", file=sys.stderr)

    raw_windows = {}
    for filename in WINDOW_FILES:
        path = SCRIPT_DIR / filename
        if not path.exists():
            print(f"  [skip] {filename}: not found", file=sys.stderr)
            continue
        df = load_window(filename).dropna(subset=FEATURE_COLUMNS)
        if df.empty:
            print(f"  [skip] {filename}: 0 usable rows", file=sys.stderr)
            continue
        raw_windows[filename] = df
        print(f"  {filename}: {len(df)} rows", file=sys.stderr)

    # Leave-one-window-out: score window i using a classifier trained
    # on every OTHER window concatenated together -- see this module's
    # own docstring for why the single all-4-combined saved model can't
    # be reused here without leaking training rows into the evaluation.
    windows = {}
    for held_out in raw_windows:
        train_df = pd.concat([df for name, df in raw_windows.items() if name != held_out], ignore_index=True)
        y_train = train_df["realized_label"].astype(str).to_numpy()
        X_train = train_df[FEATURE_COLUMNS].astype("float64").to_numpy()
        model = _fit_model(model_name, build_secondary_model(), X_train, y_train)
        windows[held_out] = compute_scores(raw_windows[held_out], model)

    print(f"\n{'ML_WEIGHT':>10}" + "".join(f"{name.replace('ml_training_set_ticker_universe_', '').replace('.csv', ''):>14}" for name in windows) + f"{'POOLED':>10}", file=sys.stderr)
    print("-" * (10 + 14 * len(windows) + 10), file=sys.stderr)

    pooled_by_weight = {}
    for ml_weight in ML_WEIGHTS:
        row_str = f"{ml_weight:>10.2f}"
        pooled_correct = 0
        pooled_scored = 0
        for name, df in windows.items():
            outcome = evaluate(df, ml_weight)
            if outcome is None:
                row_str += f"{'--':>14}"
                continue
            correct, scored, acc = outcome
            row_str += f"{acc:>13.1f}%"
            pooled_correct += correct
            pooled_scored += scored
        pooled_acc = 100 * pooled_correct / pooled_scored if pooled_scored else None
        pooled_by_weight[ml_weight] = pooled_acc
        row_str += f"{pooled_acc:>9.1f}%" if pooled_acc is not None else f"{'--':>10}"
        print(row_str, file=sys.stderr)

    best_weight = max(pooled_by_weight, key=lambda w: pooled_by_weight[w])
    baseline_acc = pooled_by_weight[0.0]
    print(f"\nBest pooled ML_WEIGHT: {best_weight} -> {pooled_by_weight[best_weight]:.1f}% "
          f"(vs. {baseline_acc:.1f}% at ML_WEIGHT=0, i.e. today's production config)", file=sys.stderr)

    # Plateau check, same spirit as tune_recommendation_config.py's own
    # near_best count -- a robust improvement should hold up across
    # several nearby weights, not spike at one specific value.
    near_best = [w for w, acc in pooled_by_weight.items() if pooled_by_weight[best_weight] - acc <= 1.0]
    print(f"{len(near_best)} of {len(ML_WEIGHTS)} tested weights are within 1 point of the best "
          f"({'looks like a robust plateau' if len(near_best) >= 4 else 'looks like a narrow spike -- treat with caution'}).")


if __name__ == "__main__":
    main()
