from app.valuation.dcf_engine import DCFEngine
from app.valuation.fcff_engine import FCFFEngine
from app.valuation.wacc_engine import WACCEngine
from app.valuation.sensitivity_engine import SensitivityEngine

class ValuationPipeline:
  def __init__(self,financial_df,market_cap,beta):
    self.financial_df=(financial_df)
    self.market_cap=(market_cap)
    self.beta=(beta)
  
  def run_valuation(self):
    fcff_engine=(FCFFEngine(self.financial_df))
    revenue_forecasts=fcff_engine.forecast_revenue()
    fcff_forecasts=fcff_engine.forecast_fcff()
    wacc_engine=(WACCEngine(financial_df=self.financial_df,market_cap=self.market_cap,beta=self.beta))

    wacc=(wacc_engine.calculate_wacc())

    dcf_engine=(DCFEngine(forecast_fcff_df=fcff_forecasts,discount_rate=wacc))

    enterprise_value=dcf_engine.calculate_enterprise_value()

    sensitvity_engine=(SensitivityEngine(enterprise_value))
    sensivity_analysis=(sensitvity_engine.generate_matrix)
    equity_value=(dcf_engine.calculate_equity_value(total_debt=self.financial_df["total_debt"].iloc[-1],cash=self.financial_df["cash"].iloc[-1]))
    
    return {
    "enterprise_value": enterprise_value,
    "equity_value": equity_value,
    "fcff_forecasts": fcff_forecasts,
    "wacc": wacc,
    "sensivity_analysis":sensivity_analysis
    }