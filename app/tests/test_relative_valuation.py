"""
Unit tests for RelativeValuationEngine (app/valuation/relative_valuation.py)
-- compares a company's current EV/EBITDA (or EV/Revenue fallback)
against ITS OWN historical average, hand-computed against small
fixtures so the signal boundaries (NEUTRAL_BAND_PCT=10.0, matching the
precedent in alpha_factors.py) can be checked exactly, not just
range-checked.

No network, no models -- pure pandas math against small hand-built
fixtures.
"""

import pandas as pd
import pytest

from app.valuation.relative_valuation import RelativeValuationEngine


# Three fiscal years, flat EBITDA (ebit=8 + depreciation=2 = 10) and
# flat shares (100), zero debt/cash so EV == price * shares exactly.
# Year-end prices of 5.0 for the first two (historical) years give a
# clean historical average EV/EBITDA of (500/10 + 500/10) / 2 = 50.0.
_INDEX = pd.to_datetime(["2022-12-31", "2023-12-31", "2024-12-31"])


def _ev_ebitda_df(**overrides):
    base = {
        "ebit": [8, 8, 8], "depreciation": [2, 2, 2],
        "total_debt": [0, 0, 0], "cash": [0, 0, 0],
        "shares_outstanding": [100, 100, 100],
    }
    base.update(overrides)
    return pd.DataFrame(base, index=_INDEX)


def _prices(**overrides):
    return pd.DataFrame({"Close": [5.0, 5.0, 6.0]}, index=_INDEX)


# ---------------------------------------------------------------- EV/EBITDA signal boundaries

def test_signal_is_expensive_when_current_multiple_is_well_above_history():
    # market_cap=555 -> current EV/EBITDA = 555/10 = 55.5, historical avg
    # 50.0 -> +11% vs history, clearly past the +10% NEUTRAL_BAND_PCT edge.
    engine = RelativeValuationEngine(_ev_ebitda_df(), _prices(), market_cap=555, current_price=None)
    result = engine.evaluate()
    assert result["metric"] == "EV/EBITDA"
    assert result["current_ev_ebitda"] == pytest.approx(55.5)
    assert result["historical_avg_ev_ebitda"] == pytest.approx(50.0)
    assert result["vs_history_pct"] == pytest.approx(11.0)
    assert result["signal"] == "expensive"


def test_signal_is_cheap_when_current_multiple_is_well_below_history():
    # market_cap=445 -> current EV/EBITDA = 44.5 -> -11% vs history.
    engine = RelativeValuationEngine(_ev_ebitda_df(), _prices(), market_cap=445, current_price=None)
    result = engine.evaluate()
    assert result["vs_history_pct"] == pytest.approx(-11.0)
    assert result["signal"] == "cheap"


def test_signal_is_in_line_just_inside_the_positive_neutral_band():
    # market_cap=549 -> +9.8% vs history, just short of the +10% edge.
    engine = RelativeValuationEngine(_ev_ebitda_df(), _prices(), market_cap=549, current_price=None)
    result = engine.evaluate()
    assert result["vs_history_pct"] == pytest.approx(9.8)
    assert result["signal"] == "in-line"


def test_signal_is_in_line_just_inside_the_negative_neutral_band():
    # market_cap=451 -> -9.8% vs history, just short of the -10% edge.
    engine = RelativeValuationEngine(_ev_ebitda_df(), _prices(), market_cap=451, current_price=None)
    result = engine.evaluate()
    assert result["vs_history_pct"] == pytest.approx(-9.8)
    assert result["signal"] == "in-line"


def test_historical_average_excludes_the_most_recent_fiscal_year():
    # The historical average must be built from years 1-2 only (year 3's
    # ev_ebitda -- 660/10=66 at price 6.0 -- must NOT be pulled into the
    # average, only used for the standalone "current" side of the ratio
    # via self.market_cap).
    engine = RelativeValuationEngine(_ev_ebitda_df(), _prices(), market_cap=555, current_price=None)
    result = engine.evaluate()
    assert result["historical_avg_ev_ebitda"] == pytest.approx(50.0)
    assert result["years_used"] == 2


# ---------------------------------------------------------------- EV/Revenue fallback

def test_falls_back_to_ev_revenue_when_ebitda_is_negative_every_year():
    # ebit=-8, depreciation=2 -> ebitda=-6 every year -> EV/EBITDA
    # can't form a signal (Phase 2 backtesting gap the docstring
    # describes for unprofitable growth companies) -- falls back to
    # EV/Revenue. Historical EV/Revenue = (500/50 + 500/50)/2 = 10.0;
    # current = 530/60 = 8.83 -> -11.67% -> cheap.
    df = _ev_ebitda_df(ebit=[-8, -8, -8], revenue=[50, 50, 60])
    engine = RelativeValuationEngine(df, _prices(), market_cap=530, current_price=None)
    result = engine.evaluate()

    assert result["metric"] == "EV/Revenue"
    assert "EV/EBITDA can't form a signal" in result["method"]
    assert result["current_ev_ebitda"] == pytest.approx(8.83, abs=0.01)  # reuses EBITDA's field names, see module docstring
    assert result["historical_avg_ev_ebitda"] == pytest.approx(10.0)
    assert result["signal"] == "cheap"


def test_prefers_ev_ebitda_over_ev_revenue_when_both_are_available():
    # Positive EBITDA every year -> EV/EBITDA is tried first and wins,
    # even though revenue is also present and would otherwise support
    # an EV/Revenue calculation.
    df = _ev_ebitda_df(revenue=[50, 50, 60])
    engine = RelativeValuationEngine(df, _prices(), market_cap=555, current_price=None)
    result = engine.evaluate()
    assert result["metric"] == "EV/EBITDA"


# ---------------------------------------------------------------- graceful degradation to None

def test_evaluate_returns_none_when_required_columns_are_missing():
    # No total_debt/cash/shares_outstanding/revenue at all -- neither
    # EV/EBITDA nor its EV/Revenue fallback can be computed.
    df = pd.DataFrame({"ebit": [8], "depreciation": [2]}, index=_INDEX[:1])
    engine = RelativeValuationEngine(df, _prices(), market_cap=555, current_price=None)
    assert engine.evaluate() is None


def test_evaluate_returns_none_when_historical_prices_are_missing():
    engine = RelativeValuationEngine(_ev_ebitda_df(), None, market_cap=555, current_price=None)
    assert engine.evaluate() is None


def test_evaluate_returns_none_when_historical_prices_are_empty():
    engine = RelativeValuationEngine(_ev_ebitda_df(), pd.DataFrame(), market_cap=555, current_price=None)
    assert engine.evaluate() is None


def test_evaluate_returns_none_when_market_cap_is_missing():
    engine = RelativeValuationEngine(_ev_ebitda_df(), _prices(), market_cap=None, current_price=None)
    assert engine.evaluate() is None


def test_evaluate_returns_none_with_fewer_than_two_fiscal_years():
    df = _ev_ebitda_df().iloc[:1]
    engine = RelativeValuationEngine(df, _prices().iloc[:1], market_cap=555, current_price=None)
    assert engine.evaluate() is None


# ---------------------------------------------------------------- P/FCF cross-check

def test_pfcf_cross_check_is_attached_when_available():
    df = _ev_ebitda_df(cash_from_operations=[20, 20, 20], capex=[5, 5, 5])
    engine = RelativeValuationEngine(df, _prices(), market_cap=555, current_price=15.0)
    result = engine.evaluate()
    # fcf_per_share = (20-5)/100 = 0.15 -> pfcf = price / fcf_per_share
    # Values are rounded to 2dp by the engine, so compare with matching precision.
    assert result["current_pfcf"] == pytest.approx(15.0 / 0.15, abs=0.01)
    assert result["historical_avg_pfcf"] == pytest.approx(5.0 / 0.15, abs=0.01)


def test_pfcf_cross_check_is_absent_when_cash_flow_columns_are_missing():
    engine = RelativeValuationEngine(_ev_ebitda_df(), _prices(), market_cap=555, current_price=15.0)
    result = engine.evaluate()
    assert "current_pfcf" not in result
    assert "historical_avg_pfcf" not in result


# ---------------------------------------------------------------- timezone handling

def test_tz_aware_historical_prices_are_handled_without_crashing():
    tz_prices = _prices().copy()
    tz_prices.index = tz_prices.index.tz_localize("UTC")
    engine = RelativeValuationEngine(_ev_ebitda_df(), tz_prices, market_cap=555, current_price=None)
    result = engine.evaluate()
    assert result is not None
    assert result["signal"] == "expensive"
