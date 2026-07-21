"""
what_if_dcf.py

Cheap, single-shot DCF recompute for the Streamlit "what-if" sliders
panel. Not a new calculation path: reuses the exact same
FCFFEngine.forecast_fcff() override-params mechanism already built for
Monte Carlo sampling (app/valuation/monte_carlo_dcf.py) and the same
DCFEngine used everywhere else in the app -- just run once per slider
move instead of thousands of times. No LLM call, no network fetch, no
Monte Carlo loop, so this is sub-second by construction (it's one
FCFFEngine.forecast_fcff() + one DCFEngine pass, the same cost as a
single Monte Carlo sample).

Composite score / rating deliberately import report_data_builder.py's
own _dcf_score/_composite_score/_rating_from_score rather than
re-deriving the formula here -- a second implementation of the same
math is exactly the kind of duplication that has drifted and caused
real bugs earlier this session (see the NaN-to-+100 composite-scoring
bug). Importing the "private" (underscore-prefixed) functions directly
keeps the what-if verdict mathematically identical to the real
recommendation's formula by construction, not by two implementations
happening to agree.
"""

from typing import Optional

from app.valuation.dcf_engine import DCFEngine
from app.valuation.fcff_engine import FCFFEngine
from app.reporting.report_data_builder import _dcf_score, _composite_score, _rating_from_score

# Kept in sync with valuation_pipeline.DEFAULT_TERMINAL_GROWTH_RATE --
# see that constant's comment for why 4%, not 3%.
DEFAULT_TERMINAL_GROWTH_RATE = 0.04
DEFAULT_FORECAST_YEARS = 10

# Slider bounds. WACC's bounds are NOT here -- they're centered on
# each company's own computed WACC (matching the sensitivity grid's
# own WACC axis, see valuation_pipeline.SENSITIVITY_WACC_OFFSETS) and
# computed by the caller. Terminal growth reuses
# valuation_pipeline.SENSITIVITY_GROWTH_RANGE's span. Growth rate has
# no equivalent in the sensitivity grid (that grid only varies WACC x
# terminal growth) -- reuses monte_carlo_dcf.GROWTH_RATE_CLIP, the
# next-closest already-established bound for this exact parameter.
GROWTH_RATE_MIN, GROWTH_RATE_MAX = -0.10, 0.30
TERMINAL_GROWTH_MIN, TERMINAL_GROWTH_MAX = 0.01, 0.05
WACC_OFFSET = 0.02  # +/- around the company's own raw WACC


def compute_what_if(
    financial_df,
    total_debt: float,
    cash: float,
    shares_outstanding: float,
    current_price: float,
    growth_rate: float,
    wacc: float,
    terminal_growth_rate: float,
    relative_score: Optional[float],
    forecast_years: int = DEFAULT_FORECAST_YEARS,
) -> Optional[dict]:
    """
    Recomputes intrinsic value / upside / composite score / rating
    for one (growth_rate, wacc, terminal_growth_rate) combination.
    relative_score is passed in, not recomputed -- the sliders only
    perturb DCF assumptions; relative valuation is a separate signal
    that doesn't change when the user drags a DCF slider.

    Returns None if the base FCFF can't be computed at all (mirrors
    ValuationPipeline's own "DCF doesn't apply here" case -- the
    what-if panel simply shouldn't be shown for those companies,
    same as the real report skips DCF for them).
    """
    fcff_engine = FCFFEngine(financial_df)
    base_fcff = fcff_engine.calculate_normalized_base_fcff()
    if base_fcff is None or base_fcff <= 0:
        return None

    forecast = fcff_engine.forecast_fcff(
        forecast_years=forecast_years,
        terminal_growth_rate=terminal_growth_rate,
        base_fcff_override=base_fcff,
        initial_growth_rate_override=growth_rate,
    )
    dcf = DCFEngine(forecast_fcff_df=forecast, discount_rate=wacc, terminal_growth_rate=terminal_growth_rate)
    equity_value = dcf.calculate_equity_value(total_debt=total_debt, cash=cash)
    intrinsic_value = equity_value / shares_outstanding
    upside_percent = (intrinsic_value - current_price) / current_price * 100

    dcf_score = _dcf_score(upside_percent)
    composite_score = _composite_score(dcf_score, relative_score)
    rating = _rating_from_score(composite_score)

    wacc_info = dcf.wacc_floor_info()

    return {
        "intrinsic_value": intrinsic_value,
        "upside_percent": upside_percent,
        "dcf_score": dcf_score,
        "relative_score": relative_score,
        "composite_score": composite_score,
        "rating": rating,
        "wacc_used": wacc_info["wacc_used"],
        "wacc_floored": wacc_info["floored"],
    }
