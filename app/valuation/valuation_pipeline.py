import pandas as pd

from app.valuation.dcf_engine import DCFEngine
from app.valuation.fcff_engine import FCFFEngine
from app.valuation.wacc_engine import WACCEngine
from app.valuation.sensitivity_analysis import SensitivityAnalysis
from app.valuation.monte_carlo_dcf import MonteCarloDCFEngine

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

# Terminal growth assumptions to sweep in the sensitivity table. Kept
# fixed across companies (a reasonable long-run real-growth range),
# unlike the WACC axis which is centered on the company's own computed
# WACC below.
SENSITIVITY_GROWTH_RANGE = [0.01, 0.02, 0.03, 0.04, 0.05]
SENSITIVITY_WACC_OFFSETS = [-0.02, -0.01, 0.0, 0.01, 0.02]


class ValuationPipeline:
  def __init__(self,financial_df,market_cap,beta):
    self.financial_df=(financial_df)
    self.market_cap=(market_cap)
    self.beta=(beta)

  def run_valuation(self):
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
            f"negative (${base_fcff/1e6:,.0f}M) -- this company's working "
            f"capital consumption structurally exceeds its profitability, "
            f"so a discounted cash flow model cannot produce a meaningful "
            f"intrinsic value here. The recommendation below is instead "
            f"derived from relative valuation and sentiment."
        )

    revenue_forecasts=fcff_engine.forecast_revenue()
    fcff_forecasts=fcff_engine.forecast_fcff(terminal_growth_rate=DEFAULT_TERMINAL_GROWTH_RATE)
    wacc_engine=(WACCEngine(financial_df=self.financial_df,market_cap=self.market_cap,beta=self.beta))

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
        terminal_growth_rate=DEFAULT_TERMINAL_GROWTH_RATE,
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
            base_terminal_growth=DEFAULT_TERMINAL_GROWTH_RATE,
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
    "terminal_growth_rate": DEFAULT_TERMINAL_GROWTH_RATE,
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
          "terminal_growth_rate": DEFAULT_TERMINAL_GROWTH_RATE,
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
