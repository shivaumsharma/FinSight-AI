import pandas as pd 
import numpy as np 
from app.valuation.dcf_engine import DCFEngine

class SensitivityAnalysis:
  def __init__(self,forecast_fcff_df,total_debt,cash,shares_outstanding):
    self.forecast_fcff_df=(forecast_fcff_df)
    self.total_debt=(total_debt)
    self.cash=(cash)
    self.shares_outstanding=(shares_outstanding)
  
  def generate_sensitivity_table(self,wacc_range,growth_range):
    sensitivity_table=pd.DataFrame(index=wacc_range,columns=growth_range)

    for wacc in wacc_range:
      for growth_rate in growth_range:
        dcf_engine=DCFEngine(forecast_fcff_df=self.forecast_fcff_df,discount_rate=wacc,terminal_growth_rate=growth_rate)

        intrinsic_value=(dcf_engine.calculate_intrinsic_value(total_debt=self.total_debt,cash=self.cash,shares_outstanding=self.shares_outstanding))

        sensitivity_table.loc[wacc,growth_rate]=intrinsic_value

    return sensitivity_table

    