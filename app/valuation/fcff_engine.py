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


    def calculate_capex_to_revenue_ratio(self):
        revenue=(self.financial_df["revenue"].dropna())
        capex=(self.financial_df["capex"].dropna())
        aligned_df=pd.concat([revenue,capex],axis=1).dropna()
        aligned_df.columns=["revenue","capex"]
        aligned_df=aligned_df[aligned_df["revenue"]>0]
        return aligned_df["capex"]/aligned_df["revenue"]

    def calculate_normalized_capex(self):
        """
        Multi-year average capex/revenue ratio applied to current
        revenue, instead of using the latest year's actual capex.
        A single year of unusually heavy capex (e.g. an AI datacenter
        buildout) or unusually light capex distorts a base FCFF that
        uses that year's raw capex figure directly; normalizing capex
        specifically -- while still using current-year revenue, EBIT,
        and change in NWC as-is -- isolates the one line item that's
        actually lumpy instead of smoothing all of FCFF (which was
        also dragging down NOPAT/NWC contributions from good years).
        """
        ratio=(self.calculate_capex_to_revenue_ratio())
        average_ratio=(ratio.mean())
        current_revenue=(self.financial_df["revenue"].dropna().iloc[-1])
        return average_ratio*current_revenue

    def _raw_change_in_nwc(self):
        """
        Change in NWC WITHOUT calculate_change_in_nwc()'s first-year
        zero-fill. That fill exists so calculate_fcff() has a usable
        (non-NaN) value for the first historical year even though
        there's no prior year to diff against -- but a manufactured
        zero is not a real data point, and including it here would
        bias a ratio *average* toward zero. Used only for normalization.
        """
        return self.calculate_nwc().dropna().diff()

    def calculate_nwc_to_revenue_ratio(self):
        revenue=(self.financial_df["revenue"].dropna())
        change_in_nwc=(self._raw_change_in_nwc())
        aligned_df=pd.concat([revenue,change_in_nwc],axis=1).dropna()
        aligned_df.columns=["revenue","change_in_nwc"]
        aligned_df=aligned_df[aligned_df["revenue"]>0]
        return aligned_df["change_in_nwc"]/aligned_df["revenue"]

    def calculate_normalized_change_in_nwc(self):
        """
        Multi-year average change-in-NWC/revenue ratio applied to
        current revenue -- the same normalization already applied to
        capex, extended to ΔNWC. A single year's working-capital swing
        (e.g. a one-time inventory build, receivables timing, or an
        M&A-driven balance sheet shift) is exactly as capable of
        distorting the forecast base as a one-off capex spike; using
        it as-is reintroduces the same single-year-outlier fragility
        the capex fix was meant to solve, just on a different line
        item.
        """
        ratio=(self.calculate_nwc_to_revenue_ratio())
        average_ratio=(ratio.mean())
        current_revenue=(self.financial_df["revenue"].dropna().iloc[-1])
        return average_ratio*current_revenue

    def calculate_normalized_base_fcff(self):
        """
        Base year FCFF built from components, not a straight mean of
        historical total FCFF. Current-year NOPAT (current EBIT and
        that year's own tax rate) and current-year depreciation are
        used as-is. Capex and change in NWC are both normalized (see
        calculate_normalized_capex / calculate_normalized_change_in_nwc)
        since both are line items that can produce one-off,
        non-representative swings in a single year.

        Returns None (not a crash) when a required line item is
        structurally absent for this company's reporting format --
        e.g. a bank: "EBIT", "capex", and a current/non-current asset
        split are non-financial-company concepts that don't apply to
        how banks report, so ebit/capex/current_assets/current_liabilities
        come back entirely NaN rather than merely missing a recent
        year. FCFF-DCF cannot run at all for that case (a different
        situation from the negative-but-computable base FCFF that
        triggers the "Insufficient Data" fallback elsewhere) -- the
        caller (ValuationPipeline) treats None the same way as a
        negative base: skip DCF, don't crash.
        """
        nopat_series=(self.calculate_nopat().dropna())
        depreciation_series=(self.financial_df["depreciation"].dropna())
        if nopat_series.empty or depreciation_series.empty:
            return None

        normalized_capex=(self.calculate_normalized_capex())
        normalized_change_in_nwc=(self.calculate_normalized_change_in_nwc())
        if pd.isna(normalized_capex) or pd.isna(normalized_change_in_nwc):
            return None

        nopat_current=nopat_series.iloc[-1]
        depreciation_current=depreciation_series.iloc[-1]

        return nopat_current+depreciation_current-normalized_capex-normalized_change_in_nwc

    def forecast_fcff(self,forecast_years=10,terminal_growth_rate=0.03,
                       base_fcff_override=None,initial_growth_rate_override=None):
       """
       base_fcff_override / initial_growth_rate_override: let a caller
       substitute a sampled value for the normally-computed base FCFF
       / revenue CAGR while still running the real three-stage fade
       and terminal-growth logic below -- used by MonteCarloDCFEngine
       (app/valuation/monte_carlo_dcf.py) to perturb growth assumptions
       per-sample without duplicating this method's fade math. None
       (the default) preserves the original deterministic behavior for
       every existing caller.
       """
       base_fcff=(base_fcff_override if base_fcff_override is not None else self.calculate_normalized_base_fcff())

       initial_growth_rate=(initial_growth_rate_override if initial_growth_rate_override is not None else self.calculate_revenue_cagr())

       # Three-stage fade, not a single linear fade across the whole
       # window. A high-growth mega-cap doesn't decelerate to terminal
       # growth in a uniform straight line over just 5 years -- that
       # structurally caps its implied value at roughly a mature-company
       # multiple regardless of how strong the business actually is.
       # Years 1-3 hold at the company's current growth rate, years 4-7
       # fade linearly to an intermediate rate (the midpoint between
       # current and terminal growth), and years 8-10 fade linearly from
       # that intermediate rate down to terminal_growth_rate, reaching it
       # exactly by the final year. This gives a genuine high-growth
       # company more of the explicit forecast window to compound at a
       # rate closer to its real trajectory before the Gordon-growth
       # terminal value (anchored to the final year) takes over.
       intermediate_growth_rate=(initial_growth_rate+terminal_growth_rate)/2

       stage1_years=3
       stage2_years=4
       stage3_years=forecast_years-stage1_years-stage2_years

       forecasted_fcff=[]
       current_fcff=base_fcff

       for year in range(1,forecast_years+1):
           if year<=stage1_years:
               year_growth_rate=initial_growth_rate
           elif year<=stage1_years+stage2_years:
               progress=(year-stage1_years)/stage2_years
               year_growth_rate=(
                   initial_growth_rate
                   +(intermediate_growth_rate-initial_growth_rate)*progress
               )
           else:
               progress=(year-stage1_years-stage2_years)/stage3_years
               year_growth_rate=(
                   intermediate_growth_rate
                   +(terminal_growth_rate-intermediate_growth_rate)*progress
               )

           current_fcff = current_fcff * (1 + year_growth_rate)
           forecasted_fcff.append(current_fcff)

       forecast_index = [f"Year_{i}" for i in range(1,forecast_years + 1)]
       forecast_df = pd.DataFrame({"forecast_fcff":forecasted_fcff},index=forecast_index)

       return forecast_df
           

