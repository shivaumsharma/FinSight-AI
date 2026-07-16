from app.valuation.dcf_engine import DCFEngine
from app.valuation.fcff_engine import FCFFEngine
from app.valuation.wacc_engine import WACCEngine
from app.valuation.sensitivity_analysis import SensitivityAnalysis

DEFAULT_TERMINAL_GROWTH_RATE = 0.03

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
    revenue_forecasts=fcff_engine.forecast_revenue()
    fcff_forecasts=fcff_engine.forecast_fcff(terminal_growth_rate=DEFAULT_TERMINAL_GROWTH_RATE)
    wacc_engine=(WACCEngine(financial_df=self.financial_df,market_cap=self.market_cap,beta=self.beta))

    wacc=(wacc_engine.calculate_wacc())

    dcf_engine=(DCFEngine(
        forecast_fcff_df=fcff_forecasts,
        discount_rate=wacc,
        terminal_growth_rate=DEFAULT_TERMINAL_GROWTH_RATE,
    ))

    enterprise_value=dcf_engine.calculate_enterprise_value()

    total_debt = self.financial_df["total_debt"].iloc[-1]
    cash = self.financial_df["cash"].iloc[-1]

    equity_value=(dcf_engine.calculate_equity_value(total_debt=total_debt,cash=cash))

    sensitivity_analysis = self._build_sensitivity_table(
        fcff_forecasts=fcff_forecasts,
        base_wacc=wacc,
        total_debt=total_debt,
        cash=cash,
    )

    return {
    "enterprise_value": enterprise_value,
    "equity_value": equity_value,
    "fcff_forecasts": fcff_forecasts,
    "wacc": wacc,
    "terminal_growth_rate": DEFAULT_TERMINAL_GROWTH_RATE,
    "sensitivity_analysis":sensitivity_analysis
    }

  def _build_sensitivity_table(self, fcff_forecasts, base_wacc, total_debt, cash):
      """
      Re-runs the actual DCF model across a WACC x terminal-growth grid
      via SensitivityAnalysis, instead of approximating enterprise value
      with a formula. The WACC axis is centered on this company's own
      computed WACC and floored above the highest growth assumption,
      since the Gordon growth terminal-value formula is undefined (or
      goes negative) once WACC <= terminal growth rate.
      """
      shares_outstanding = None
      if "shares_outstanding" in self.financial_df.columns:
          series = self.financial_df["shares_outstanding"].dropna()
          if not series.empty:
              shares_outstanding = series.iloc[-1]

      if not shares_outstanding:
          return None

      max_growth = max(SENSITIVITY_GROWTH_RANGE)
      wacc_range = sorted({
          round(max(base_wacc + offset, max_growth + 0.01), 4)
          for offset in SENSITIVITY_WACC_OFFSETS
      })

      return SensitivityAnalysis(
          forecast_fcff_df=fcff_forecasts,
          total_debt=total_debt,
          cash=cash,
          shares_outstanding=shares_outstanding,
      ).generate_sensitivity_table(wacc_range, SENSITIVITY_GROWTH_RANGE)
