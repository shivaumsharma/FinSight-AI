import pandas as pd 

METRIC_MAPPINGS={
  "revenue":[
    "Total Revenue",
    "Operating Revenue",
    "Revenue"

  ],

  "ebit":[
    "EBIT",
    "Operating Income",
    "Operating Income Loss"
  ],

  "net_income":[
    "Net Income",
    "Net Income Common Stockholders"
  ],

  "cash_from_operations":[
    "Operating Cash Flow",
    "Cash Flow From Continuing Operating Activities"
  ],

  "capex":[
    "Capital Expenditure",
    "Capital Expenditures",
    "Purchase Of PPE",
    "Net PPE Purchase And Sale"
  ],

  "total_debt":[
    "Total Debt",
    "Long Term Debt"
  ],

  "tax_expense":[
      "Tax Provision",
      "Provision For Income Taxes"
  ],
  
  "pretax_income":[
      "Pretax Income",
      "Pre Tax Income"
  ],

  "depreciation":[
      "Depreciation",
      "Depreciation And Amortization",
      "Depreciation Ammortizations Depletion"

  ],

  "current_assets":[
      "Current Assets"
  ],

  "current_liabilities":[
    "Current Liabilities"
  ],

  "cash":[
      "Cash And Cash Equivalents",
      "Cash Cash Equivalents And Short Term Investments"
  ],

  "shares_outstanding":[
      "Ordinary Shares Number",
      "Share Issued",
      "Common Stock Shares Outstanding"
  ]
}

class FinancialStatementNormaliser:
    def __init__(self,income_statement,balance_sheet,cash_flow):
        
        self.income_statement=income_statement
        self.balance_sheet=balance_sheet
        self.cash_flow=cash_flow

    def extract_metric(self,statement_df,aliases):
       for alias in aliases:
           if alias in statement_df.index:
               metric_series=statement_df.loc[alias]
               metric_series=metric_series.dropna()

               return metric_series
       return pd.Series(dtype=float)
    
    def normalise(self):
        
        normalised_data={}
        normalised_data["revenue"]=self.extract_metric(self.income_statement,METRIC_MAPPINGS["revenue"])
        normalised_data["ebit"]=self.extract_metric(self.income_statement,METRIC_MAPPINGS["ebit"])
        normalised_data["net_income"]=self.extract_metric(self.income_statement,METRIC_MAPPINGS["net_income"])
        normalised_data["cash_from_operations"]=self.extract_metric(self.cash_flow,METRIC_MAPPINGS["cash_from_operations"])
        normalised_data["capex"]=self.extract_metric(self.cash_flow,METRIC_MAPPINGS["capex"])
        normalised_data["total_debt"]=self.extract_metric(self.balance_sheet,METRIC_MAPPINGS["total_debt"])
        normalised_data["tax_expense"]=(self.extract_metric(self.income_statement,METRIC_MAPPINGS["tax_expense"]))
        normalised_data["pretax_income"]=(self.extract_metric(self.income_statement,METRIC_MAPPINGS["pretax_income"]))
        normalised_data["depreciation"]=(self.extract_metric(self.cash_flow,METRIC_MAPPINGS["depreciation"]))
        normalised_data["current_assets"]=(self.extract_metric(self.balance_sheet,METRIC_MAPPINGS["current_assets"]))
        normalised_data["current_liabilities"]=(self.extract_metric(self.balance_sheet,METRIC_MAPPINGS["current_liabilities"]))
        normalised_data["cash"]=(self.extract_metric(self.balance_sheet,METRIC_MAPPINGS["cash"]))
        normalised_data["shares_outstanding"]=(self.extract_metric(self.balance_sheet,METRIC_MAPPINGS["shares_outstanding"]))

        normalised_df=pd.DataFrame(normalised_data)
        cleaned_df=self.align_and_clean(normalised_df)

        return cleaned_df
    
    def align_and_clean(self,normalised_df):
        
        normalised_df.index=pd.to_datetime(
            normalised_df.index
        )
        normalised_df=normalised_df.sort_index()

        normalised_df=normalised_df.apply(pd.to_numeric,errors="coerce")

        if "capex" in normalised_df.columns:
          normalised_df["capex"] = (normalised_df["capex"].abs())

        normalised_df=normalised_df.dropna(how="all")
    
        return normalised_df

    def validate_financials(self,df):
        warnings=[]
        if "revenue" in df.columns:
            if(df["revenue"]<=0).any():
                warnings.append("Negative or zero value detected")

        if "cash_from_operations" in df.columns:
            if(df["cash_from_operations"]<=0).all():
                warnings.append("operating cash flows are negative")
        
        return warnings
    

from app.data.market_data import MarketDataLoader

if __name__ == "__main__":
  loader=MarketDataLoader("AAPL")
  income_stml=loader.get_income_statement()
  balance_sheet=loader.get_balance_sheet()
  cash_flow=loader.get_cash_flow()

  normaliser=FinancialStatementNormaliser(income_stml,balance_sheet,cash_flow)
  normalised_data=normaliser.normalise()

  print(normalised_data)

  print("\n===CASH FLOW INDEX===")
  print(cash_flow.index)

  print("\n===BALANCE SHEET INDEX===")
  print(balance_sheet.index)

  warnings=normaliser.validate_financials(normalised_data)

  print("\n=== VALIDATION WARNINGS ===")
  for warning in warnings:
      print(warning)
