"""
ml_features.py

Flat feature vector for the ML valuation classifier
(ml_valuation_classifier.py) -- single source of truth for feature
construction, used both to build the training set
(scripts/build_ml_training_set.py) and at live inference time
(ValuationTool), so there is no train/serve skew between the two.

Feature *concept set* was adapted from a separate, read-only-audited
project (the DCF Valuation Engine) -- see the integration audit --
but every value here is read from FinSight's own valuation_results /
relative_valuation / monte_carlo output, not recomputed via that
project's code (its underlying DCF math was found materially weaker:
no capital-structure-weighted WACC, no capex/NWC outlier
normalization, a `net_income * 0.7` FCF fallback).

One addition beyond that project's feature set:
`relative_vs_history_pct`, sourced from relative_valuation.py -- a
genuinely independent signal (this company's own multiple vs. its
trading history) that project never had.
"""

from typing import Any, Dict, Optional

from app.core.research_context import ResearchContext
from app.valuation.fcff_engine import FCFFEngine

FEATURE_COLUMNS = [
    "growth_rate",
    "wacc",
    "beta",
    "revenue_growth",
    "fcf_yield",
    "dcf_over_price",
    "mc_mean_over_price",
    "mc_std_over_price",
    "mc_prob_undervalued",
    "net_cash_per_share_over_price",
    "relative_vs_history_pct",
]


def extract_features(context: ResearchContext) -> Optional[Dict[str, Any]]:
    """
    Returns a flat dict of FEATURE_COLUMNS values, or None if DCF
    wasn't available for this company or a required input is missing
    -- the ML classifier, like the recommendation composite, only
    applies where DCF actually ran (see Insufficient Data handling
    elsewhere in the pipeline; this classifier doesn't get its own
    separate fallback).
    """
    valuation_results = context.valuation_results or {}
    if not valuation_results.get("dcf_available"):
        return None

    intrinsic_value = valuation_results.get("intrinsic_value")
    current_price = valuation_results.get("current_price")
    if intrinsic_value is None or not current_price:
        return None

    financial_df = context.normalized_financials
    if financial_df is None or financial_df.empty:
        return None

    required_cols = ("shares_outstanding", "total_debt", "cash", "revenue")
    if not all(c in financial_df.columns for c in required_cols):
        return None

    shares_series = financial_df["shares_outstanding"].dropna()
    debt_series = financial_df["total_debt"].dropna()
    cash_series = financial_df["cash"].dropna()
    revenue_series = financial_df["revenue"].dropna()
    if shares_series.empty or debt_series.empty or cash_series.empty or len(revenue_series) < 2:
        return None

    shares_outstanding = shares_series.iloc[-1]
    total_debt = debt_series.iloc[-1]
    cash = cash_series.iloc[-1]

    fcff_engine = FCFFEngine(financial_df)
    growth_rate = fcff_engine.calculate_revenue_cagr()
    base_fcff = fcff_engine.calculate_normalized_base_fcff()
    if base_fcff is None:
        return None

    prev_revenue = revenue_series.iloc[-2]
    revenue_growth = (revenue_series.iloc[-1] - prev_revenue) / prev_revenue if prev_revenue else 0.0

    net_cash_per_share = (cash - total_debt) / shares_outstanding
    fcf_per_share = base_fcff / shares_outstanding

    monte_carlo = valuation_results.get("monte_carlo") or {}
    relative = valuation_results.get("relative_valuation") or {}

    return {
        "growth_rate": growth_rate,
        "wacc": valuation_results.get("wacc"),
        "beta": context.beta,
        "revenue_growth": revenue_growth,
        "fcf_yield": fcf_per_share / current_price,
        "dcf_over_price": intrinsic_value / current_price,
        "mc_mean_over_price": (monte_carlo["mean"] / current_price) if monte_carlo else None,
        "mc_std_over_price": (monte_carlo["std_dev"] / current_price) if monte_carlo else None,
        "mc_prob_undervalued": monte_carlo.get("prob_undervalued"),
        "net_cash_per_share_over_price": net_cash_per_share / current_price,
        "relative_vs_history_pct": relative.get("vs_history_pct"),
    }
