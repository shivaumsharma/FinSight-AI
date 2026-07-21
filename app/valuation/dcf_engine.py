import pandas as pd 
import numpy as np 

class DCFEngine:

  # Minimum WACC-to-terminal-growth spread, in absolute terms (e.g.
  # 0.03 = 3 percentage points). The Gordon-growth terminal value has
  # a 1/(discount_rate - terminal_growth_rate) term -- once that
  # spread gets small, terminal value (and therefore the whole DCF
  # output) blows up non-linearly and becomes numerically unstable
  # rather than reflecting a genuine valuation signal. Applied here,
  # inside the engine itself, so every caller -- the primary DCF in
  # ValuationPipeline AND every cell of the sensitivity grid (each of
  # which constructs its own DCFEngine) -- gets identical treatment
  # through the same code path, instead of two places independently
  # reimplementing (and risking drifting from) the same stability rule.
  #
  # Was 0.04, lowered to 0.03 in the same change that raised
  # DEFAULT_TERMINAL_GROWTH_RATE 3%->4% (see valuation_pipeline.py).
  # The floor is terminal_growth + this spread, so raising terminal
  # growth alone would have also raised the floor itself for every
  # already-floored company (confirmed by hand: JNJ's WACC floor moved
  # 7%->8%, which discounts its cash flows MORE heavily and made its
  # valuation WORSE, the opposite of the terminal-growth change's
  # intent). Lowering the spread by the same 1 point keeps the floor
  # at exactly 7% for those companies (unaffected), while companies
  # whose raw WACC already clears the floor still get the full benefit
  # of the higher terminal growth.
  MIN_WACC_TERMINAL_SPREAD = 0.03

  def __init__(self,forecast_fcff_df,discount_rate=0.10,terminal_growth_rate=0.03):
    self.forecast_fcff_df=(forecast_fcff_df)
    self.terminal_growth_rate=(terminal_growth_rate)

    # The raw, beta-implied WACC is preserved (never silently
    # discarded) so callers can report both what was actually
    # computed and what was used for the calculation.
    self.raw_discount_rate=(discount_rate)

    floor=(terminal_growth_rate+self.MIN_WACC_TERMINAL_SPREAD)
    if discount_rate-terminal_growth_rate<self.MIN_WACC_TERMINAL_SPREAD:
        self.discount_rate=floor
        self.wacc_floored=True
    else:
        self.discount_rate=discount_rate
        self.wacc_floored=False

  def wacc_floor_info(self):
      return {
          "raw_wacc":self.raw_discount_rate,
          "wacc_used":self.discount_rate,
          "floored":self.wacc_floored,
      }

  def calculate_discount_factors(self):
      num_years=len(self.forecast_fcff_df)

      discount_factors=[]

      for year in range(1,num_years+1):
        discount_factor=(1/((1+self.discount_rate)**year))

        discount_factors.append(discount_factor)
      return pd.Series(discount_factors,index=self.forecast_fcff_df.index)
    
  def discount_fcff(self):
      discount_factors=(self.calculate_discount_factors())
      forecast_fcff=(self.forecast_fcff_df["forecast_fcff"])

      discounted_fcff=(forecast_fcff*discount_factors)

      return pd.DataFrame({
        "forecast_fcff":forecast_fcff,
        "discount_factor":discount_factors,
        "discounted_fcff":discounted_fcff
      })
    
  def calculate_terminal_value(self):
      final_fcff=(self.forecast_fcff_df["forecast_fcff"].iloc[-1])
      terminal_fcff=(final_fcff*(1+self.terminal_growth_rate))
      terminal_value=(terminal_fcff/(self.discount_rate-self.terminal_growth_rate))
      return terminal_value 
    
  def discount_terminal_value(self):
      terminal_value=(self.calculate_terminal_value())
      num_years=len(self.forecast_fcff_df)
      discounted_terminal_value=(terminal_value/(1+self.discount_rate)**num_years) 
      return discounted_terminal_value
    
  def calculate_enterprise_value(self):
      discount_fcff_df=(self.discount_fcff())
      pv_fcff=(discount_fcff_df["discounted_fcff"].sum())
      discount_terminal_value=(self.discount_terminal_value())
      enterprise_value=(pv_fcff+discount_terminal_value)
      return enterprise_value
    
  def calculate_equity_value(self,total_debt,cash):
      enterprise_value=(self.calculate_enterprise_value())
      net_debt=(total_debt-cash)
      equity_value=(enterprise_value-net_debt)
      return equity_value
    
  def calculate_intrinsic_value(self,shares_outstanding,total_debt,cash):
      equity_value=(self.calculate_equity_value(total_debt,cash))
      intrinsic_value=equity_value/shares_outstanding
      return intrinsic_value