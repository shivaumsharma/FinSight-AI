"""
Unit tests for AlphaFactorsEngine (app/analysis/alpha_factors.py) --
covers each factor's normal case, guard conditions, and degrade-to-None
behavior, plus the two structural guarantees this feature depends on:
individual factors degrade independently (one missing input never blanks
sibling factors, in the same or a different category), and .NS tickers
are gated off the three factors that would otherwise mix currencies/
markets (Relative Strength vs Index, Sector Relative Performance,
Interest Rate Sensitivity).
"""

import json

import pandas as pd
import pytest

from app.analysis.alpha_factors import AlphaFactorsEngine
from app.core.research_context import ResearchContext


def _financials(**cols):
    n = len(next(iter(cols.values())))
    index = pd.to_datetime([f"20{20+i}-12-31" for i in range(n)])
    return pd.DataFrame(cols, index=index)


def _prices(closes, start="2020-01-02"):
    index = pd.bdate_range(start=start, periods=len(closes))
    return pd.DataFrame({"Close": closes}, index=index)


def _engine(
    financials=None,
    prices=None,
    beta=None,
    company_info=None,
    financial_summary=None,
    sentiment_summary=None,
    news_sentiment_summary=None,
    is_non_us_listing=False,
    benchmark_history=None,
    sector_history=None,
    rate_proxy_history=None,
):
    return AlphaFactorsEngine(
        normalized_financials=financials,
        historical_prices=prices,
        beta=beta,
        company_info=company_info or {},
        financial_summary=financial_summary or {},
        sentiment_summary=sentiment_summary or {},
        news_sentiment_summary=news_sentiment_summary or {},
        is_non_us_listing=is_non_us_listing,
        benchmark_history=benchmark_history,
        sector_history=sector_history,
        rate_proxy_history=rate_proxy_history,
    )


# ---------------------------------------------------------------------
# Financial
# ---------------------------------------------------------------------

def test_financial_factors_cross_reference_financial_summary():
    engine = _engine(financial_summary={
        "Revenue CAGR (%)": 12.5,
        "Revenue Growth (%)": 8.0,
        "EBIT Growth (%)": 5.0,
        "Net Income Growth (%)": 6.0,
        "FCF Growth (%)": 4.0,
        "Operating Margin": 20.0,
        "Net Margin": 15.0,
    })
    factors = engine._financial_factors()
    assert factors["Revenue CAGR (%)"] == 12.5
    assert factors["Revenue Growth YoY (%)"] == 8.0
    assert factors["EBIT Growth YoY (%)"] == 5.0
    assert factors["Net Income Growth YoY (%)"] == 6.0
    assert factors["FCF Growth YoY (%)"] == 4.0
    assert factors["Operating Margin (%)"] == 20.0
    assert factors["Net Margin (%)"] == 15.0


def test_financial_factors_treat_unavailable_string_sentinel_as_none():
    engine = _engine(financial_summary={"Revenue CAGR (%)": "Unavailable"})
    assert engine._financial_factors()["Revenue CAGR (%)"] is None


def test_ebitda_margin_computed_from_latest_fiscal_year():
    df = _financials(revenue=[100, 200], ebit=[10, 30], depreciation=[5, 10])
    factors = _engine(financials=df)._financial_factors()
    assert factors["EBITDA Margin (%)"] == pytest.approx((30 + 10) / 200 * 100)


def test_ebitda_margin_none_when_revenue_zero():
    df = _financials(revenue=[0], ebit=[10], depreciation=[5])
    assert _engine(financials=df)._financial_factors()["EBITDA Margin (%)"] is None


def test_fcf_margin_from_cross_referenced_dollar_values():
    engine = _engine(financial_summary={"Revenue": 1000.0, "Free Cash Flow": 150.0})
    assert engine._financial_factors()["FCF Margin (%)"] == pytest.approx(15.0)


# ---------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------

def test_roa_and_roce_computed_from_latest_fiscal_year():
    df = _financials(
        total_assets=[1000, 2000], net_income=[50, 100],
        ebit=[80, 160], current_liabilities=[200, 400],
    )
    factors = _engine(financials=df)._quality_factors()
    assert factors["Return on Assets (%)"] == pytest.approx(100 / 2000 * 100)
    assert factors["Return on Capital Employed (%)"] == pytest.approx(160 / (2000 - 400) * 100)


def test_current_ratio_and_interest_coverage_handle_negative_convention():
    # yfinance-style negative-convention interest expense -- must be
    # abs()'d, not assumed positive (see wacc_engine.py's own
    # calculate_cost_of_debt for the same convention).
    df = _financials(
        current_assets=[300], current_liabilities=[150],
        ebit=[100], interest_expense=[-20],
    )
    factors = _engine(financials=df)._quality_factors()
    assert factors["Current Ratio"] == pytest.approx(2.0)
    assert factors["Interest Coverage Ratio"] == pytest.approx(5.0)


def test_quality_factors_degrade_individually_when_total_assets_missing():
    # No total_assets column at all -- ROA/ROCE must be None, but
    # Current Ratio (doesn't need total_assets) must still compute.
    df = _financials(current_assets=[300], current_liabilities=[150])
    factors = _engine(financials=df)._quality_factors()
    assert factors["Return on Assets (%)"] is None
    assert factors["Current Ratio"] == pytest.approx(2.0)


def test_piotroski_f_score_all_nine_signals_positive():
    df = _financials(
        net_income=[10, 20], total_assets=[1000, 1100],
        cash_from_operations=[8, 25], total_debt=[400, 300],
        current_assets=[200, 300], current_liabilities=[150, 150],
        shares_outstanding=[100, 100], ebit=[50, 88], revenue=[500, 605],
    )
    score = _engine(financials=df)._piotroski_f_score()
    assert score == 9
    # A genuine Python int, not numpy.int64 -- `== 9` alone doesn't
    # catch this (numpy defines __eq__ to compare by value across
    # types), but numpy.int64 is NOT JSON-serializable by FastAPI's
    # default encoder, unlike numpy.float64 (which subclasses float).
    # Confirmed via a real live run: sum() of the 9 numpy.bool_
    # comparisons (curr/prior come from a pandas row, so these are
    # numpy scalars even when every test fixture in this file uses
    # plain Python int/float literals -- pandas stores them as a numpy
    # dtype column regardless) silently returned numpy.int64, which
    # every test here comparing with `== 9`/`== 0` never caught.
    assert type(score) is int


def test_piotroski_f_score_all_nine_signals_negative():
    df = _financials(
        net_income=[20, -3], total_assets=[1000, 1000],
        cash_from_operations=[25, -5], total_debt=[300, 500],
        current_assets=[300, 100], current_liabilities=[150, 150],
        shares_outstanding=[100, 150], ebit=[88, 20], revenue=[605, 400],
    )
    assert _engine(financials=df)._piotroski_f_score() == 0


def test_piotroski_f_score_none_when_fewer_than_two_years():
    df = _financials(
        net_income=[10], total_assets=[1000], cash_from_operations=[8],
        total_debt=[400], current_assets=[200], current_liabilities=[150],
        shares_outstanding=[100], ebit=[50], revenue=[500],
    )
    assert _engine(financials=df)._piotroski_f_score() is None


def test_piotroski_f_score_none_when_a_required_column_is_missing():
    df = _financials(net_income=[10, 20], total_assets=[1000, 1100])
    assert _engine(financials=df)._piotroski_f_score() is None


def test_altman_z_score_safe_zone():
    df = _financials(
        current_assets=[500], current_liabilities=[200], total_assets=[1000],
        retained_earnings=[300], ebit=[150], total_equity=[600], revenue=[800],
    )
    engine = _engine(financials=df, company_info={"market_cap": 2000})
    z = engine._altman_z_score()
    expected = 1.2 * 0.3 + 1.4 * 0.3 + 3.3 * 0.15 + 0.6 * 5.0 + 1.0 * 0.8
    assert z == pytest.approx(expected, abs=0.01)
    assert engine._altman_zone(z) == "Safe Zone"


def test_altman_z_score_distress_zone():
    df = _financials(
        current_assets=[100], current_liabilities=[200], total_assets=[1000],
        retained_earnings=[-200], ebit=[10], total_equity=[100], revenue=[300],
    )
    engine = _engine(financials=df, company_info={"market_cap": 150})
    z = engine._altman_z_score()
    expected = 1.2 * (-0.1) + 1.4 * (-0.2) + 3.3 * 0.01 + 0.6 * (150 / 900) + 1.0 * 0.3
    assert z == pytest.approx(expected, abs=0.01)
    assert engine._altman_zone(z) == "Distress Zone"


def test_altman_z_score_none_when_retained_earnings_missing():
    df = _financials(
        current_assets=[500], current_liabilities=[200], total_assets=[1000],
        ebit=[150], total_equity=[600], revenue=[800],
    )
    engine = _engine(financials=df, company_info={"market_cap": 2000})
    assert engine._altman_z_score() is None


def test_altman_z_score_none_for_bank_shaped_financials():
    # Same structural gap DCF already has for banks -- ebit/current
    # assets/current liabilities come back NaN, not just absent.
    df = _financials(
        current_assets=[None, None], current_liabilities=[None, None],
        total_assets=[1000, 1100], retained_earnings=[300, 320],
        ebit=[None, None], total_equity=[600, 620], revenue=[800, 820],
    )
    engine = _engine(financials=df, company_info={"market_cap": 2000})
    assert engine._altman_z_score() is None


# ---------------------------------------------------------------------
# Valuation -- P/E and P/B vs the company's own trading history
# ---------------------------------------------------------------------

_PE_PB_DATES = pd.to_datetime(["2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31"])
_YEAR_END_PRICES = pd.DataFrame({"Close": [10, 15, 21, 30]}, index=_PE_PB_DATES)


def test_pe_vs_own_history_hand_computed():
    df = pd.DataFrame(
        {"net_income": [50, 60, 70, 80], "shares_outstanding": [100, 100, 100, 100]},
        index=_PE_PB_DATES,
    )
    engine = _engine(
        financials=df, prices=_YEAR_END_PRICES,
        company_info={"current_price": 40},
        financial_summary={"EPS": 0.8},
    )
    result = engine._valuation_factors()["P/E vs Own History"]
    # Yearly P/E: 10/0.5=20, 15/0.6=25, 21/0.7=30, 30/0.8=37.5
    # historical_avg excludes the latest year -> mean([20, 25, 30]) = 25
    # current = 40 / 0.8 = 50 -> vs_history_pct = (50/25 - 1) * 100 = 100%
    assert result["current"] == pytest.approx(50.0)
    assert result["historical_avg"] == pytest.approx(25.0)
    assert result["years_used"] == 3
    assert result["vs_history_pct"] == pytest.approx(100.0)
    assert result["signal"] == "expensive"


def test_pb_vs_own_history_hand_computed():
    df = pd.DataFrame(
        {"total_equity": [400, 500, 600, 700], "shares_outstanding": [100, 100, 100, 100]},
        index=_PE_PB_DATES,
    )
    engine = _engine(
        financials=df, prices=_YEAR_END_PRICES,
        company_info={"current_price": 40},
    )
    result = engine._valuation_factors()["P/B vs Own History"]
    # Yearly P/B: 10/4=2.5, 15/5=3.0, 21/6=3.5, 30/7=4.2857
    # historical_avg = mean([2.5, 3.0, 3.5]) = 3.0
    # current bvps = 700/100 = 7 -> current = 40/7 -- every field is
    # rounded to 2dp by the module (matching relative_valuation.py's own
    # convention), so expectations below are pre-rounded too.
    assert result["current"] == pytest.approx(round(40 / 7, 2))
    assert result["historical_avg"] == pytest.approx(3.0)
    assert result["vs_history_pct"] == pytest.approx(round((40 / 7 / 3.0 - 1) * 100, 2))
    assert result["signal"] == "expensive"


def test_pe_vs_own_history_skips_negative_eps_years():
    df = pd.DataFrame(
        {"net_income": [-5, 60, 70, 80], "shares_outstanding": [100, 100, 100, 100]},
        index=_PE_PB_DATES,
    )
    engine = _engine(
        financials=df, prices=_YEAR_END_PRICES,
        company_info={"current_price": 40},
        financial_summary={"EPS": 0.8},
    )
    result = engine._valuation_factors()["P/E vs Own History"]
    # 2020's negative EPS (-0.05) is excluded entirely -- not counted as
    # a zero, not crashing. Valid yearly P/E: 2021=15/0.6=25, 2022=21/0.7=30,
    # 2023=30/0.8=37.5. historical_avg excludes the latest (2023) ->
    # mean([25, 30]) = 27.5, so only 2 (not 3) years feed the average.
    assert result is not None
    assert result["years_used"] == 2
    assert result["historical_avg"] == pytest.approx(27.5)
    assert result["current"] == pytest.approx(50.0)  # 40 / 0.8


def test_pe_vs_own_history_none_when_current_eps_non_positive():
    df = pd.DataFrame(
        {"net_income": [50, 60, 70, 80], "shares_outstanding": [100, 100, 100, 100]},
        index=_PE_PB_DATES,
    )
    engine = _engine(
        financials=df, prices=_YEAR_END_PRICES,
        company_info={"current_price": 40},
        financial_summary={"EPS": -1.0},
    )
    assert engine._valuation_factors()["P/E vs Own History"] is None


# ---------------------------------------------------------------------
# Market -- pure historical_prices, plus benchmark-gated factors
# ---------------------------------------------------------------------

def test_market_factors_full_window():
    closes = list(range(100, 400))  # 300 trading days, prices 100..399
    prices = _prices(closes)
    factors = _engine(prices=prices)._market_factors()
    window = pd.Series(closes).tail(252)
    high, low, current = window.max(), window.min(), closes[-1]
    assert factors["Price vs 52-Week High (%)"] == pytest.approx((current / high - 1) * 100, abs=0.01)
    assert factors["Price vs 52-Week Low (%)"] == pytest.approx((current / low - 1) * 100, abs=0.01)
    assert factors["12-Month Price Momentum (%)"] == pytest.approx(
        (closes[-1] / closes[-252] - 1) * 100, abs=0.01
    )
    assert factors["6-Month Price Momentum (%)"] == pytest.approx(
        (closes[-1] / closes[-126] - 1) * 100, abs=0.01
    )


def test_market_factors_52_week_fields_none_when_under_a_year_of_history():
    closes = list(range(100, 250))  # 150 trading days: enough for 6mo, not 12mo
    prices = _prices(closes)
    factors = _engine(prices=prices)._market_factors()
    assert factors["Price vs 52-Week High (%)"] is None
    assert factors["12-Month Price Momentum (%)"] is None
    assert factors["6-Month Price Momentum (%)"] is not None


def test_market_factors_gated_off_for_ns_tickers_even_with_benchmark_supplied():
    closes = list(range(100, 400))
    prices = _prices(closes)
    benchmark = _prices([200] * 300)
    factors = _engine(prices=prices, benchmark_history=benchmark, is_non_us_listing=True)._market_factors()
    assert factors["Relative Strength vs Index (%)"] is None
    assert factors["Sector Relative Performance (%)"] is None
    assert "_market_note" in factors


def test_relative_strength_vs_index_computed_for_us_tickers():
    stock_closes = list(range(100, 400))
    benchmark_closes = [200] * 300  # flat benchmark -> 0% benchmark return
    prices = _prices(stock_closes)
    benchmark = _prices(benchmark_closes)
    factors = _engine(prices=prices, benchmark_history=benchmark, is_non_us_listing=False)._market_factors()
    stock_6m = (stock_closes[-1] / stock_closes[-126] - 1) * 100
    assert factors["Relative Strength vs Index (%)"] == pytest.approx(stock_6m - 0.0, abs=0.01)


# ---------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------

def test_beta_passthrough():
    assert _engine(beta=1.35)._risk_factors()["Beta"] == 1.35


def test_annualized_volatility_none_under_a_year_of_history():
    prices = _prices(list(range(100, 200)))
    assert _engine(prices=prices)._risk_factors()["Annualized Volatility (%)"] is None


def test_annualized_volatility_positive_for_a_full_year_of_history():
    closes = [100 + (i % 5) for i in range(260)]  # noisy but bounded series
    prices = _prices(closes)
    vol = _engine(prices=prices)._risk_factors()["Annualized Volatility (%)"]
    assert vol is not None and vol > 0


def test_risk_factors_include_altman_zone():
    df = _financials(
        current_assets=[500], current_liabilities=[200], total_assets=[1000],
        retained_earnings=[300], ebit=[150], total_equity=[600], revenue=[800],
    )
    factors = _engine(financials=df, company_info={"market_cap": 2000})._risk_factors()
    assert factors["Altman Z-Score Zone"] == "Safe Zone"


# ---------------------------------------------------------------------
# Sentiment (pure cross-reference)
# ---------------------------------------------------------------------

def test_sentiment_factors_cross_reference_existing_summaries():
    factors = _engine(
        sentiment_summary={"Overall Sentiment": "Positive"},
        news_sentiment_summary={"Overall Sentiment": "Neutral"},
    )._sentiment_factors()
    assert factors["Management/SEC Filing Sentiment"] == "Positive"
    assert factors["News/Media Sentiment"] == "Neutral"


# ---------------------------------------------------------------------
# Macro
# ---------------------------------------------------------------------

def test_interest_rate_sensitivity_none_for_ns_tickers():
    prices = _prices(list(range(100, 400)))
    rate_proxy = _prices([4.0 + (i % 3) * 0.01 for i in range(300)])
    factors = _engine(prices=prices, rate_proxy_history=rate_proxy, is_non_us_listing=True)._macro_factors()
    assert factors["Interest Rate Sensitivity"] is None
    assert "_macro_note" in factors


def test_interest_rate_sensitivity_computed_for_us_tickers_with_enough_data():
    prices = _prices([100 + (i % 7) for i in range(300)])
    rate_proxy = _prices([4.0 + (i % 5) * 0.02 for i in range(300)])
    factors = _engine(prices=prices, rate_proxy_history=rate_proxy, is_non_us_listing=False)._macro_factors()
    assert factors["Interest Rate Sensitivity"] is not None
    assert "_interest_rate_sensitivity_definition" in factors


def test_interest_rate_sensitivity_none_when_rate_proxy_missing():
    prices = _prices(list(range(100, 400)))
    factors = _engine(prices=prices, is_non_us_listing=False)._macro_factors()
    assert factors["Interest Rate Sensitivity"] is None


# ---------------------------------------------------------------------
# Cross-category degradation -- the structural guarantee the whole
# feature depends on: a corrupted/missing input degrades only the
# factor(s) that actually need it, never sibling factors in the same
# or a different category.
# ---------------------------------------------------------------------

def test_missing_total_debt_only_affects_factors_that_need_it():
    df = _financials(
        current_assets=[300], current_liabilities=[150],
        net_income=[50], total_assets=[1000], ebit=[80],
        # total_debt deliberately omitted
    )
    engine = _engine(
        financials=df,
        sentiment_summary={"Overall Sentiment": "Positive"},
        news_sentiment_summary={"Overall Sentiment": "Neutral"},
    )
    result = engine.evaluate()

    # Piotroski needs total_debt -- must degrade to None.
    assert result["quality"]["Piotroski F-Score (0-9, proxy)"] is None
    # Current Ratio doesn't need total_debt -- must still compute.
    assert result["quality"]["Current Ratio"] == pytest.approx(2.0)
    # An entirely unrelated category must be fully unaffected.
    assert result["sentiment"]["Management/SEC Filing Sentiment"] == "Positive"
    assert result["sentiment"]["News/Media Sentiment"] == "Neutral"


def test_evaluate_returns_all_seven_categories():
    result = AlphaFactorsEngine(
        normalized_financials=None,
        historical_prices=None,
        beta=None,
        company_info={},
        financial_summary={},
        sentiment_summary={},
        news_sentiment_summary={},
        is_non_us_listing=False,
    ).evaluate()
    assert set(result.keys()) == {
        "financial", "quality", "valuation", "market", "risk", "sentiment", "macro",
    }


# ---------------------------------------------------------------------
# Full-pipeline integration -- ValuationTool.run() actually wires
# AlphaFactorsEngine in and attaches its output to
# context.valuation_results["alpha_factors"]. Constructs a bare
# ResearchContext by hand (matching test_api.py's own precedent,
# test_normalized_financials_round_trip_is_exact), pre-populating
# normalized_financials so ValuationTool doesn't try to run
# MarketDataTool (a real network call) first.
# ---------------------------------------------------------------------

def _multi_year_financials():
    dates = pd.to_datetime(["2022-12-31", "2023-12-31", "2024-12-31"])
    return pd.DataFrame(
        {
            "revenue": [800.0, 900.0, 1000.0],
            "ebit": [150.0, 170.0, 190.0],
            "net_income": [100.0, 115.0, 130.0],
            "cash_from_operations": [120.0, 140.0, 160.0],
            "capex": [20.0, 22.0, 25.0],
            "total_debt": [300.0, 280.0, 260.0],
            "tax_expense": [25.0, 28.0, 30.0],
            "pretax_income": [125.0, 143.0, 160.0],
            "depreciation": [30.0, 32.0, 35.0],
            "current_assets": [400.0, 450.0, 500.0],
            "current_liabilities": [200.0, 210.0, 220.0],
            "cash": [150.0, 180.0, 210.0],
            "shares_outstanding": [100.0, 100.0, 100.0],
            "interest_expense": [-15.0, -14.0, -13.0],
            "total_equity": [600.0, 650.0, 700.0],
            "total_assets": [1200.0, 1300.0, 1400.0],
            "retained_earnings": [400.0, 450.0, 500.0],
        },
        index=dates,
    )


def test_valuation_tool_wires_alpha_factors_onto_the_context(monkeypatch):
    from app.tools import valuation_tool as vt_module
    from app.tools.valuation_tool import ValuationTool

    # ValuationTool.run() fetches the benchmark/sector/rate-proxy series
    # for a non-.NS ticker -- a real yfinance network call this test
    # must NOT make (this exact class of bug -- a "unit" test silently
    # depending on live network -- is what broke CI for
    # test_company_resolver.py; see .github/workflows/tests.yml's own
    # comment on that). Stub it out entirely.
    monkeypatch.setattr(vt_module, "get_benchmark_history", lambda ticker, period="5y": None)

    context = ResearchContext(ticker="AAPL", question="Should I invest in AAPL?")
    context.normalized_financials = _multi_year_financials()
    context.historical_prices = _prices([50 + i * 0.1 for i in range(300)])
    context.market_cap = 8_200_000_000
    context.beta = 1.2
    context.company_info = {"current_price": 82.0, "sector": "Technology", "market_cap": 8_200_000_000}

    ValuationTool().run(context)

    alpha_factors = context.valuation_results.get("alpha_factors")
    assert alpha_factors is not None
    assert set(alpha_factors.keys()) == {
        "financial", "quality", "valuation", "market", "risk", "sentiment", "macro",
    }
    # At least the zero-new-fetch, structurally-complete factors should
    # have actually computed, not silently degraded to None across the
    # board -- a real assertion that the wiring passes real data through,
    # not just that the dict shape is right.
    assert alpha_factors["quality"]["Current Ratio"] is not None
    assert alpha_factors["risk"]["Beta"] == 1.2

    # The regression that actually broke production (caught only by a
    # real live run, not by any of the value-correctness assertions
    # above): json.dumps must succeed on the whole scorecard, the same
    # serialization path FastAPI's response encoder exercises. Piotroski's
    # int(sum(...)) fix (see alpha_factors.py) is what this proves stays
    # fixed -- str(default=str) below is deliberately NOT used, since
    # that would silently paper back over a numpy.int64 regression
    # instead of failing on it.
    json.dumps(alpha_factors)
