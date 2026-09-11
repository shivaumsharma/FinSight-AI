import pandas as pd 
import numpy as np 


class WACCEngine:

  # risk_free_rate/market_risk_premium default to a US CAPM approximation
  # (~10Y Treasury yield, ~long-run US equity risk premium). Callers with
  # a non-USD company should pass currency-appropriate values instead of
  # relying on these defaults -- see valuation_pipeline.py's
  # RISK_FREE_RATE_BY_CURRENCY / MARKET_RISK_PREMIUM_BY_CURRENCY for the
  # only place in this codebase that currently does so.
  def  __init__(self,financial_df,market_cap,beta,risk_free_rate=0.04,market_risk_premium=0.06):
    self.financial_df=financial_df
    self.market_cap=market_cap
    self.beta=beta
    self.risk_free_rate=risk_free_rate
    self.market_risk_premium=market_risk_premium
    

  def calculate_cost_of_equity(self):
    cost_of_equity=(self.risk_free_rate+(self.beta*self.market_risk_premium))
    return cost_of_equity

  def calculate_cost_of_debt(self):
    interest_expense=(self.financial_df["interest_expense"].dropna())
    total_debt=(self.financial_df["total_debt"].dropna())
    aligned_df=pd.concat([interest_expense,total_debt],axis=1).dropna()
    aligned_df.columns=["interest_expense","total_debt"]
    aligned_df=aligned_df[aligned_df["total_debt"]>0]
    aligned_df["cost_of_debt"]=(aligned_df["interest_expense"].abs()/aligned_df["total_debt"])    
    average_cost_of_debt=(aligned_df["cost_of_debt"].mean())

    return average_cost_of_debt
  
  def calculate_debt_value(self):
    total_debt=(self.financial_df["total_debt"].dropna())
    if total_debt.empty:
      # Structurally absent debt history (every reported year is NaN)
      # is treated the same as "this company has no debt" -- 0.0, not
      # a crash. calculate_wacc() below already has to handle a
      # genuinely debt-free company the same way, so there's no
      # meaningful downstream difference between "known to be debt-free"
      # and "debt data was never reported" as far as this engine is
      # concerned.
      return 0.0
    latest_debt_value=(total_debt.iloc[-1])
    return latest_debt_value

  
  def calculate_total_value(self):
    equity_value=(self.market_cap)
    debt_value=(self.calculate_debt_value())
    total_value=(equity_value+debt_value)
    return total_value
  
  def calculate_tax_rate(self):
    tax_expense=(self.financial_df["tax_expense"].dropna())
    pretax_income=(self.financial_df["pretax_income"].dropna())
    aligned_df=pd.concat([tax_expense,pretax_income],axis=1).dropna()
    aligned_df.columns=["tax_expense","pretax_income"]
    aligned_df=aligned_df[aligned_df["pretax_income"]>0]
    aligned_df["tax_rate"]=(aligned_df["tax_expense"]/aligned_df["pretax_income"])
    aligned_df["tax_rate"]=(aligned_df["tax_rate"].clip(0,0.5))
    average_tax_rate=(aligned_df["tax_rate"].mean())
    return average_tax_rate

  def calculate_wacc(self):
    equity_value=(self.market_cap)
    debt_value=(self.calculate_debt_value())
    total_value=(self.calculate_total_value())
    cost_of_equity=(self.calculate_cost_of_equity())
    equity_weight=(equity_value/total_value)
    debt_weight=(debt_value/total_value)
    if debt_weight<=0:
      # A genuinely debt-free company (or one with no usable debt
      # history -- calculate_debt_value() already collapses both cases
      # to debt_value=0.0) has a textbook WACC equal to its cost of
      # equity. Skip cost_of_debt/tax_rate entirely rather than
      # computing them anyway: calculate_cost_of_debt() averages over
      # total_debt>0 rows, which is NaN (mean of an empty frame) exactly
      # when there's no such row -- and 0 * NaN is NaN in IEEE-754, not
      # 0, so multiplying that NaN by a legitimately-zero debt_weight
      # used to poison the whole WACC anyway instead of cancelling out.
      return equity_weight*cost_of_equity
    cost_of_debt=(self.calculate_cost_of_debt())
    tax_rate=(self.calculate_tax_rate())
    wacc=((equity_weight*cost_of_equity)+(debt_weight*cost_of_debt*(1-tax_rate)))
    return wacc
