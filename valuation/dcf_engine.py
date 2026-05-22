import pandas as pd 
import numpy as np 

class DCFEngine:
  def __init__(self,forecast_fcff_df,discount_rate=0.10,terminal_growth_rate=0.03):
    self.forecast_fcff_df=(forecast_fcff_df)
    self.discount_rate=(discount_rate)
    self.terminal_growth_rate=(terminal_growth_rate)

  def caculate_discount_factors(self):
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
      terminal_value=(terminal_fcff/(self.dicount_rate-self.terminal_growth_rate))
      return terminal_value 
    
  def discount_terminal_value(self):
      terminal_value=(self.calculate_terminal_value())
      num_years=len(self.forecast_fcff_df)
      discounted_terminal_value=(terminal_value/(1+self.discount_rate)**num_years) 
      return discount_terminal_value
    
  def calculate_entreprise_value(self):
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
    
