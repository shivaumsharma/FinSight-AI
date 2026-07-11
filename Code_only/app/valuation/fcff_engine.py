import pandas as pd
import numpy as np 

class FCFFEngine:

    def __init__(self,financial_df):
        self.financial_df=financial_df
    
    def calculate_revenue_cagr(self):
        revenue_series=self.financial_df["revenue"].dropna()
        beginning_value=revenue_series.iloc[0]
        ending_value=revenue_series.iloc[-1]
        n=len(revenue_series)-1

        cagr=((ending_value/beginning_value)**(1/n))-1

        return cagr
    
    def forecast_revenue(self,forecast_years=5):
        revenue_series=(self.financial_df["revenue"].dropna())

        current_revenue=revenue_series.iloc[-1]

        growth_rate=(self.calculate_revenue_cagr())

        forecasted_revenues=[]

        for year in range(1,forecast_years+1):
            future_revenue=(current_revenue*(1+growth_rate)**year)
            forecasted_revenues.append(future_revenue)
        forecast_index=[f"Year_{i}" for i in range(1,forecast_years+1)]
        forecast_df=pd.DataFrame({"forecast_revenue": forecasted_revenues},index=forecast_index)
        return forecast_df


    def calculate_tax_rate(self):
        tax_expense=(self.financial_df["tax_expense"].dropna())
        pretax_income=(self.financial_df["pretax_income"].dropna())
        aligned_df=pd.concat([tax_expense,pretax_income],axis=1).dropna()
        aligned_df.columns=["tax_expense","pretax_income"]

        aligned_df["tax_rate"]=(aligned_df["tax_expense"]/aligned_df["pretax_income"])

        aligned_df["tax_rate"]=(aligned_df["tax_rate"].clip(0,0.5))

        return aligned_df["tax_rate"]
    
    def calculate_nopat(self):
        ebit=(self.financial_df["ebit"].dropna())
        tax_rate=(self.calculate_tax_rate().dropna())
        aligned_df=pd.concat([ebit,tax_rate],axis=1).dropna()
        aligned_df.columns=["ebit","tax_rate"]
        aligned_df["nopat"]=(aligned_df["ebit"]*(1-aligned_df["tax_rate"]))

        return aligned_df["nopat"]

    
    def calculate_nwc(self):
        current_assets=(self.financial_df["current_assets"].dropna())
        current_liabilities=(self.financial_df["current_liabilities"].dropna())
        aligned_df=pd.concat([current_assets,current_liabilities],axis=1).dropna()
        aligned_df.columns=["current_assets","current_liabilities"]
        aligned_df["nwc"]=aligned_df["current_assets"]-aligned_df["current_liabilities"]
        
        return aligned_df["nwc"]
    
    def calculate_change_in_nwc(self):
        nwc=(self.calculate_nwc().dropna())
        change_in_nwc=nwc.diff().fillna(0)
        
        return change_in_nwc

    def calculate_fcff(self):
        nopat=(self.calculate_nopat().dropna())
        depreciation=(self.financial_df["depreciation"].dropna())
        capex=(self.financial_df["capex"].dropna())
        change_in_nwc=(self.calculate_change_in_nwc().dropna())
        aligned_df=pd.concat([nopat,depreciation,capex,change_in_nwc],axis=1).dropna()
        aligned_df.columns=["nopat","depreciation","capex","change_in_nwc"]
        aligned_df["fcff"]=(aligned_df["nopat"]+aligned_df["depreciation"]-aligned_df["capex"]-aligned_df["change_in_nwc"])
        return aligned_df["fcff"]


    def forecast_fcff(self,forecast_years=5):
       historical_fcff=(self.calculate_fcff().dropna())
       latest_fcff=historical_fcff.iloc[-1]
       forecasted_fcff=[]
       growth_rate=(self.calculate_revenue_cagr())

       for year in range(1,forecast_years+1):
           future_fcff = ( latest_fcff *(1 + growth_rate) ** year)
           forecasted_fcff.append(future_fcff)
           forecast_index = [f"Year_{i}"for i in range(1,forecast_years + 1)]

       forecast_df = pd.DataFrame({"forecast_fcff":forecasted_fcff},index=forecast_index)

       return forecast_df
           

