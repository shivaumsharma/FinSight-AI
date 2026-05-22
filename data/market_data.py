import yfinance as yf
import pandas as pd 

class MarketDataLoader:
  
  def __init__(self,ticker:str):
     self.ticker=ticker.upper()
     self.stock=yf.Ticker(self.ticker)

  def get_company_info(self):
     info =self.stock.info
     
     return {
        "company_name":info.get("longName"),
        "sector":info.get("sector"),
        "industry":info.get("industry"),
        "market_cap":info.get("marketCap"),
        "currency":info.get("currency"),
        "country":info.get("country")
     }
  
  def get_historical_prices(self,period="5y"):
     
     df=self.stock.history(period=period)
     if df.empty:
        raise ValueError("No price data found for {self.ticker}")
     return df
  
  def get_income_statement(self):
     df=self.stock.financials

     if df.empty:
        raise ValueError("Income Statement unavailable")
     return df 
     
  
  def get_balance_sheet(self):
     df=self.stock.balance_sheet

     if df.empty:
        raise ValueError("Balance Sheet unavailable")
     return df 
  
  def get_cash_flow(self):
     df=self.stock.cashflow

     if df.empty:
        raise ValueError("Cash flow statement unavailable")
     return df 


  def extract_metric(self,statement_df,metric_name):
      if metric_name in statement_df.index:
          return statement_df.loc[metric_name]
      return None


if __name__=="__main__":
   loader=MarketDataLoader("AAPL")
   income_stml=loader.get_income_statement()
   revenue=loader.extract_metric(
      income_stml,
      "Total Revenue"
   )

   print("\n===Revenue===")
   print(revenue)
   print(income_stml.index)
   print(income_stml.columns)
   print(income_stml.shape)

   print("\n===ORIGINAL SHAPE===")
   print(income_stml.shape)

   transposed_stmt=(income_stml.T).sort_index()
   
   transposed_stmt["Revenue Growth"]=(
      transposed_stmt["Operating Revenue"]
      .pct_change()
   )

   print("\n===Revenue Growth===")
   print(
      transposed_stmt[
         ["Operating Revenue","Revenue Growth"]
      ]
   )

   revenue_series=transposed_stmt["Operating Revenue"]
   
   revenue_series=revenue_series.dropna()
   starting_value=revenue_series.iloc[0]
   ending_value=revenue_series.iloc[-1]
   n=len(revenue_series)-1
   cagr=((ending_value/starting_value)**(1/n))-1

   print("\n===CAGR===")
   print(f"{cagr:.2%}")

   