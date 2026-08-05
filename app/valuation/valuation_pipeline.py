import pandas as pd

from app.core.currency import currency_symbol
from app.valuation.dcf_engine import DCFEngine
from app.valuation.fcff_engine import FCFFEngine
from app.valuation.wacc_engine import WACCEngine
from app.valuation.sensitivity_analysis import SensitivityAnalysis
from app.valuation.monte_carlo_dcf import MonteCarloDCFEngine
from app.core.cache import cache_get, cache_set, make_key

# Content-addressed, not TTL-based for correctness (see app/core/cache.py's
# module docstring): run_valuation()'s output (WACC, FCFF forecast,
# sensitivity table, Monte Carlo distribution) is entirely derived from
# financial_df/market_cap/beta and never touches a live price -- current
# price/upside_percent are computed separately, downstream, in
# valuation_tool.py, always fresh. market_cap is rounded before hashing
# (see _round_market_cap) so a hit still requires market cap not having
# moved meaningfully, without demanding byte-for-byte equality on a
# number that technically ticks with every trade. TTL below is a
# storage bound only.
VALUATION_CACHE_TTL_SECONDS = 3600
_MARKET_CAP_ROUNDING = 1e8  # nearest $100M

# 3% (roughly long-run real GDP growth alone) was the original value.
# Backtesting (scripts/phase2_backtest.py, ~1000-ticker universe)
# showed a strong systematic bias: median DCF-implied upside across
# the whole universe was -30.8% -- more companies flagged "overvalued"
# than is plausible for a broad market index. Traced by hand (WMT:
# model says $463B enterprise value vs. $949B market-implied) to the
# terminal-value math, not the cash-flow base (WMT's own projected
# FCFF was already slightly ABOVE its reported free cash flow) --
# closing that gap fully would need an implausibly high terminal
# growth assumption, but 3% alone is on the low end of standard
# practice, which typically ties terminal growth to long-run NOMINAL
# GDP growth (real growth + inflation, historically ~4% in the US),
# not real growth alone. Raised to 4% as the first, cheapest candidate
# fix -- see scripts/phase2_backtest.py's own accuracy comparison
# before/after for whether this alone meaningfully closes the gap.
DEFAULT_TERMINAL_GROWTH_RATE = 0.04

# CAPM inputs and terminal growth, keyed by the company's reporting
# currency (from context.company_info["currency"], via yfinance) --
# WACCEngine's cost-of-equity calculation and the terminal-growth
# assumption above are both calibrated to US markets; using them
# unmodified for an INR-denominated company would silently discount
# Indian cash flows at a US risk-free rate/ERP, understating WACC and
# overstating intrinsic value.
#
# INR risk-free rate approximates the 10Y Indian G-Sec yield, which
# has run close to 7% through 2023-2025 (RBI data) -- consistently
# above the US 10Y this file's own 4% approximates. INR equity risk
# premium follows the pattern in Damodaran's India country-risk-
# premium series: India's total ERP has typically run at or somewhat
# above the US ERP, since India's higher trend growth is offset by
# added country/currency risk -- kept equal to the risk-free rate
# here as a round, defensible approximation, not a precisely
# backtested figure the way DEFAULT_TERMINAL_GROWTH_RATE above is.
#
# INR terminal growth is deliberately NOT set to India's current
# nominal GDP growth (historically 9-12%, well above the US analog
# this file's 4% approximates) -- terminal growth is a "forever"
# assumption in a Gordon-growth model, and standard DCF practice caps
# it well below any single country's current growth rate on the
# expectation that high-growth economies converge toward
# steady-state over a multi-decade horizon (and a rate too close to
# the discount rate makes the terminal-value math unstable regardless
# -- see MIN_WACC_TERMINAL_SPREAD in dcf_engine.py). 5% is a
# conservative middle estimate -- clearly above the US figure, well
# below raw current growth -- pending the same kind of empirical
# backtest (scripts/phase2_backtest.py) that justified 4% for the US.
RISK_FREE_RATE_BY_CURRENCY = {"USD": 0.04, "INR": 0.07}
MARKET_RISK_PREMIUM_BY_CURRENCY = {"USD": 0.06, "INR": 0.07}
TERMINAL_GROWTH_RATE_BY_CURRENCY = {"USD": DEFAULT_TERMINAL_GROWTH_RATE, "INR": 0.05}

# equity_value = enterprise_value - net_debt (DCFEngine.calculate_equity_value).
# For a heavily-levered company, net_debt can sit close to (or above)
# enterprise_value, making equity_value a small, unstable DIFFERENCE
# between two much larger, individually uncertain numbers -- the same
# kind of numerical instability MIN_WACC_TERMINAL_SPREAD already guards
# against for the discount-rate/terminal-growth spread (DCFEngine's own
# comment), just one step further down this same calculation, and just
# as capable of turning "we don't actually have a reliable number here"
# into a confidently-reported near-zero (or negative) intrinsic value
# that would otherwise drive a Sell call nobody should trust.
#
# Confirmed by hand across the full ~80-ticker backtest universe
# (scripts/wacc_capm_audit.py, point-in-time as of 2025-08-03):
# equity_value/enterprise_value shows a stark, unambiguous gap -- CVNA
# (-56.1%, net debt literally exceeds EV), KHC (0.9%), and HAS (2.4%)
# are the only three tickers below this, then every other one of the
# 37 remaining DCF-available tickers sits at 44.1% or higher. Neither
# KHC nor HAS has negative FCFF or an obviously broken share count
# (both were traced by hand: KHC's $19.9B debt against an $18.7B
# modeled enterprise value, HAS's $3.4B debt against a $2.8B modeled
# enterprise value) -- the DCF's own leverage-extraction step is what
# produces the near-zero result, not any one input being wrong. 15% is
# a round threshold with wide margin on both sides of that gap (max
# real failure 2.4%, min real pass 44.1%), not a tight fit to it.
MIN_EQUITY_TO_ENTERPRISE_VALUE_RATIO = 0.15

# Terminal growth assumptions to sweep in the sensitivity table. Kept
# fixed across companies (a reasonable long-run real-growth range),
# unlike the WACC axis which is centered on the company's own computed
# WACC below.
SENSITIVITY_GROWTH_RANGE = [0.01, 0.02, 0.03, 0.04, 0.05]
SENSITIVITY_WACC_OFFSETS = [-0.02, -0.01, 0.0, 0.01, 0.02]


class ValuationPipeline:
  def __init__(self,financial_df,market_cap,beta,ticker=None,currency=None):
    self.financial_df=(financial_df)
    self.market_cap=(market_cap)
    self.beta=(beta)
    # Optional -- only used to make the Redis cache key human-readable
    # (see _cache_key/make_key); no valuation math depends on it. None
    # (the default) still caches correctly, just with a less legible key.
    self.ticker=(ticker)
    # Defaults to USD (unknown/missing currency behaves exactly as
    # every ticker did before this parameter existed). Drives which
    # row of RISK_FREE_RATE_BY_CURRENCY / MARKET_RISK_PREMIUM_BY_CURRENCY
    # / TERMINAL_GROWTH_RATE_BY_CURRENCY this run uses; an unrecognized
    # currency also falls back to the USD row rather than raising, so
    # an unmapped currency degrades to today's behavior instead of
    # breaking valuation entirely.
    self.currency = currency or "USD"
    self.risk_free_rate = RISK_FREE_RATE_BY_CURRENCY.get(self.currency, RISK_FREE_RATE_BY_CURRENCY["USD"])
    self.market_risk_premium = MARKET_RISK_PREMIUM_BY_CURRENCY.get(self.currency, MARKET_RISK_PREMIUM_BY_CURRENCY["USD"])
    self.terminal_growth_rate = TERMINAL_GROWTH_RATE_BY_CURRENCY.get(self.currency, DEFAULT_TERMINAL_GROWTH_RATE)
    self.currency_symbol = currency_symbol(self.currency)

  def _cache_key(self):
    rounded_market_cap = (
        round(self.market_cap / _MARKET_CAP_ROUNDING) * _MARKET_CAP_ROUNDING
        if self.market_cap else self.market_cap
    )
    financials_fingerprint = (
        self.financial_df.to_json() if self.financial_df is not None else "none"
    )
    return make_key(
        "valuation", self.ticker or "unknown", financials_fingerprint,
        rounded_market_cap, round(self.beta or 0, 2), self.currency,
    )

  def run_valuation(self):
    cache_key = self._cache_key()
    cached = cache_get(cache_key)
    if cached is not None:
      return cached

    result = self._compute_valuation()
    cache_set(cache_key, result, ttl_seconds=VALUATION_CACHE_TTL_SECONDS)
    return result

  def _compute_valuation(self):
    fcff_engine=(FCFFEngine(self.financial_df))

    # If the normalized base FCFF is negative, FCFF-DCF doesn't apply:
    # every forecast year is a multiplicative fade off that same base,
    # so it stays negative all the way through, and the resulting
    # "intrinsic value" is a negative number wearing a dollar sign --
    # not a bearish signal, a nonsensical one. This happens for
    # companies whose working-capital consumption structurally
    # exceeds their profitability (seen in Phase 1 testing: PLTR,
    # CVNA), not from a one-off outlier year (that's what capex/ΔNWC
    # normalization already handles) -- it's a genuine trait of the
    # business, so no amount of smoothing fixes it. Rather than
    # report a Sell built on a negative intrinsic value, DCF is
    # skipped entirely here and the caller (ValuationTool) falls back
    # to relative valuation + sentiment for the recommendation.
    base_fcff=(fcff_engine.calculate_normalized_base_fcff())

    # calculate_normalized_base_fcff() returns None when a required
    # line item is structurally absent for this company's reporting
    # format -- confirmed in Phase 2 testing: banks (JPM) report no
    # EBIT, no capex, and no current/non-current asset split, since
    # none of those are non-financial-company concepts a bank's
    # statements use. That's a different situation from a negative
    # base FCFF (below), but both mean the same thing for the DCF:
    # it cannot run, so both route to the same "unavailable" path.
    if base_fcff is None:
        return self._unavailable_result(
            "FCFF-DCF does not apply: one or more required line items "
            "(EBIT, capex, or a current/non-current asset split) are not "
            "reported for this company -- this is typical for banks and "
            "other financial institutions, whose statements don't use "
            "those non-financial-company concepts. The recommendation "
            "below is instead derived from relative valuation and sentiment."
        )

    if base_fcff<=0:
        return self._unavailable_result(
            f"FCFF-DCF does not apply: normalized base free cash flow is "
            f"negative ({self.currency_symbol}{base_fcff/1e6:,.0f}M) -- this company's working "
            f"capital consumption structurally exceeds its profitability, "
            f"so a discounted cash flow model cannot produce a meaningful "
            f"intrinsic value here. The recommendation below is instead "
            f"derived from relative valuation and sentiment."
        )

    revenue_forecasts=fcff_engine.forecast_revenue()
    fcff_forecasts=fcff_engine.forecast_fcff(terminal_growth_rate=self.terminal_growth_rate)
    wacc_engine=(WACCEngine(
        financial_df=self.financial_df,
        market_cap=self.market_cap,
        beta=self.beta,
        risk_free_rate=self.risk_free_rate,
        market_risk_premium=self.market_risk_premium,
    ))

    raw_wacc=(wacc_engine.calculate_wacc())

    # calculate_wacc() can come back NaN -- confirmed in backtest
    # review: CRM reports no interest_expense at all, so
    # calculate_cost_of_debt() averages an empty series to NaN, which
    # propagates through to WACC. NaN must be caught explicitly here:
    # DCFEngine's WACC floor compares `discount_rate - terminal_growth
    # < MIN_WACC_TERMINAL_SPREAD`, and NaN comparisons are always
    # False in Python, so the floor silently does NOT catch this --
    # NaN would sail through into the DCF and, downstream, into
    # derive_recommendation's composite scoring, where
    # max(-100, min(100, nan)) resolves to +100 (the same NaN-vs-`<`
    # quirk) instead of propagating -- silently turning "we don't
    # actually know" into "maximally confident Buy." Caught at the
    # root here so it can never reach either downstream quirk.
    if pd.isna(raw_wacc):
        return self._unavailable_result(
            "FCFF-DCF does not apply: WACC could not be computed -- this "
            "company reports no usable interest expense data, so the cost "
            "of debt component of WACC is unknown. The recommendation "
            "below is instead derived from relative valuation and sentiment."
        )

    dcf_engine=(DCFEngine(
        forecast_fcff_df=fcff_forecasts,
        discount_rate=raw_wacc,
        terminal_growth_rate=self.terminal_growth_rate,
    ))

    # The engine floors the WACC actually used internally if raw_wacc
    # sits too close to terminal growth for a stable Gordon-growth
    # calculation (see DCFEngine.MIN_WACC_TERMINAL_SPREAD) -- both the
    # raw and floored values are surfaced below, never silently merged.
    wacc_info = dcf_engine.wacc_floor_info()
    wacc_used = wacc_info["wacc_used"]

    enterprise_value=dcf_engine.calculate_enterprise_value()

    total_debt = self.financial_df["total_debt"].iloc[-1]
    cash = self.financial_df["cash"].iloc[-1]

    equity_value=(dcf_engine.calculate_equity_value(total_debt=total_debt,cash=cash))
    net_debt = total_debt - cash

    # See MIN_EQUITY_TO_ENTERPRISE_VALUE_RATIO's own comment above for
    # the calibration evidence. enterprise_value<=0 is checked and
    # rejected separately, not folded into one ratio comparison --
    # equity_value/enterprise_value with a NEGATIVE denominator can
    # come out positive (two negatives), which would silently evade
    # this guard exactly when the underlying number is least
    # trustworthy of all.
    if enterprise_value <= 0 or equity_value / enterprise_value < MIN_EQUITY_TO_ENTERPRISE_VALUE_RATIO:
        return self._unavailable_result(
            f"DCF is not applicable for this capital structure -- net debt ({self.currency_symbol}{net_debt:,.0f}) "
            f"leaves an equity value ({self.currency_symbol}{equity_value:,.0f}) that's under "
            f"{MIN_EQUITY_TO_ENTERPRISE_VALUE_RATIO:.0%} of the {self.currency_symbol}{enterprise_value:,.0f} "
            f"enterprise value this model computed: a small difference between two much "
            f"larger, individually uncertain numbers, not a reliable per-share value. This is "
            f"a coverage limitation of the enterprise-value DCF used here, not a data error -- "
            f"it affects any highly-levered capital structure (utilities, telecoms, REITs, "
            f"post-LBO companies), which the market typically prices by other means for "
            f"exactly this reason. The correct approach for these names is an equity DCF "
            f"(forecast FCFE directly, discount at cost of equity, skipping the enterprise-"
            f"to-equity bridge entirely) -- not implemented here."
        )

    sensitivity_analysis = self._build_sensitivity_table(
        fcff_forecasts=fcff_forecasts,
        base_wacc=raw_wacc,
        total_debt=total_debt,
        cash=cash,
    )

    wacc_floor_note = None
    if wacc_info["floored"]:
        wacc_floor_note = (
            f"WACC floored from {raw_wacc * 100:.2f}% to {wacc_used * 100:.2f}% to avoid "
            f"terminal-value instability -- raw beta-implied WACC was too close to the "
            f"terminal growth rate for a stable Gordon-growth calculation."
        )

    monte_carlo_values = None
    shares_outstanding = self._get_shares_outstanding()
    if shares_outstanding:
        monte_carlo_values = MonteCarloDCFEngine(
            fcff_engine=fcff_engine,
            base_growth_rate=fcff_engine.calculate_revenue_cagr(),
            base_wacc=wacc_used,
            base_terminal_growth=self.terminal_growth_rate,
            total_debt=total_debt,
            cash=cash,
            shares_outstanding=shares_outstanding,
        ).run()

    return {
    "dcf_available": True,
    "dcf_unavailable_reason": None,
    "enterprise_value": enterprise_value,
    "equity_value": equity_value,
    "fcff_forecasts": fcff_forecasts,
    "wacc": wacc_used,
    "raw_wacc": raw_wacc,
    "wacc_floored": wacc_info["floored"],
    "wacc_floor_note": wacc_floor_note,
    "terminal_growth_rate": self.terminal_growth_rate,
    "sensitivity_analysis":sensitivity_analysis,
    "monte_carlo_values": monte_carlo_values,
    }

  def _unavailable_result(self, reason: str) -> dict:
      """Shared shape for every 'DCF cannot run' case (negative base
      FCFF, structurally missing line items, or NaN WACC) -- every
      numeric field explicitly None so a caller can never mistake an
      absent value for a computed one."""
      return {
          "dcf_available": False,
          "dcf_unavailable_reason": reason,
          "enterprise_value": None,
          "equity_value": None,
          "fcff_forecasts": None,
          "wacc": None,
          "raw_wacc": None,
          "wacc_floored": None,
          "wacc_floor_note": None,
          "terminal_growth_rate": self.terminal_growth_rate,
          "sensitivity_analysis": None,
          "monte_carlo_values": None,
      }

  def _get_shares_outstanding(self):
      if "shares_outstanding" in self.financial_df.columns:
          series = self.financial_df["shares_outstanding"].dropna()
          if not series.empty:
              return series.iloc[-1]
      return None

  def _build_sensitivity_table(self, fcff_forecasts, base_wacc, total_debt, cash):
      """
      Re-runs the actual DCF model across a WACC x terminal-growth grid
      via SensitivityAnalysis, instead of approximating enterprise value
      with a formula. The WACC axis is centered on this company's own
      raw computed WACC (not artificially floored -- that would shift
      the whole sweep away from the company's actual implied cost of
      capital). Each individual cell is protected from a WACC/growth
      spread that's too tight by DCFEngine's own internal floor (see
      DCFEngine.MIN_WACC_TERMINAL_SPREAD), the same rule the primary
      DCF above uses -- one stability rule, applied consistently,
      instead of this method separately reimplementing its own floor.
      """
      shares_outstanding = self._get_shares_outstanding()

      if not shares_outstanding:
          return None

      wacc_range = sorted({
          round(base_wacc + offset, 4)
          for offset in SENSITIVITY_WACC_OFFSETS
      })

      return SensitivityAnalysis(
          forecast_fcff_df=fcff_forecasts,
          total_debt=total_debt,
          cash=cash,
          shares_outstanding=shares_outstanding,
      ).generate_sensitivity_table(wacc_range, SENSITIVITY_GROWTH_RANGE)
