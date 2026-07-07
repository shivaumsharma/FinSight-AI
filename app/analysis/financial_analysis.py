"""
financial_summary.py

Creates a financial intelligence summary from normalized
financial statements.

This module is responsible ONLY for financial analysis.

It does NOT perform valuation.
"""

from typing import Dict
import pandas as pd


class FinancialAnalysisBuilder:

    # ======================================================
    # Trend Interpreter
    # ======================================================
    def interpret(self, summary: dict) -> dict:
        insights = {}
        # =====================================================
        # Revenue
        # =====================================================

        growth = summary.get("Revenue Growth (%)")

        if isinstance(growth, (int, float)):

            if growth >= 20:

                insights["Revenue Analysis"] = (
                    "Revenue is growing at a very strong pace."
                )

            elif growth >= 10:

                insights["Revenue Analysis"] = (
                    "Revenue growth remains healthy."
                )

            elif growth >= 0:

                insights["Revenue Analysis"] = (
                    "Revenue growth is stable."
                )

            else:

                insights["Revenue Analysis"] = (
                    "Revenue has declined."
                )

        else:

            insights["Revenue Analysis"] = (
                "Revenue growth unavailable."
            )

        # =====================================================
        # Profitability
        # =====================================================

        margin = summary.get("Operating Margin")

        if isinstance(margin, (int, float)):

            if margin >= 30:

                insights["Profitability"] = (
                    "Operating margins are excellent."
                )

            elif margin >= 20:

                insights["Profitability"] = (
                    "Operating margins are healthy."
                )

            elif margin >= 10:

                insights["Profitability"] = (
                    "Operating margins are moderate."
                )

            else:

                insights["Profitability"] = (
                    "Operating margins are weak."
                )

        else:

            insights["Profitability"] = (
                "Profitability unavailable."
            )

        # =====================================================
        # Cash Flow
        # =====================================================

        fcf_growth = summary.get("FCF Growth (%)")

        if isinstance(fcf_growth, (int, float)):

            if fcf_growth > 10:

                insights["Cash Flow"] = (
                    "Free cash flow generation is improving."
                )

            elif fcf_growth >= 0:

                insights["Cash Flow"] = (
                    "Free cash flow is stable."
                )

            else:

                insights["Cash Flow"] = (
                    "Free cash flow is weakening."
                )

        else:

            insights["Cash Flow"] = (
                "Cash flow trend unavailable."
            )

        # =====================================================
        # Leverage
        # =====================================================

        leverage = summary.get("Debt to Equity")

        if isinstance(leverage, (int, float)):

            if leverage < 0.5:

                insights["Leverage"] = (
                    "Balance sheet leverage is low."
                )

            elif leverage < 1.5:

                insights["Leverage"] = (
                    "Leverage appears manageable."
                )

            else:

                insights["Leverage"] = (
                    "Leverage is elevated."
                )

        else:

            insights["Leverage"] = (
                "Leverage unavailable."
            )

        # =====================================================
        # ROE
        # =====================================================

        roe = summary.get("ROE")

        if isinstance(roe, (int, float)):

            if roe >= 20:

                insights["ROE Analysis"] = (
                    "Return on Equity is excellent."
                )

            elif roe >= 15:

                insights["ROE Analysis"] = (
                    "Return on Equity is healthy."
                )

            elif roe >= 10:

                insights["ROE Analysis"] = (
                    "Return on Equity is acceptable."
                )

            else:

                insights["ROE Analysis"] = (
                    "Return on Equity is weak."
                )

        else:

            insights["ROE Analysis"] = (
                "ROE unavailable."
            )

        return insights
    
    
    def _trend(self, value):

        if value == "Unavailable":
            return "Unknown"

        if value >= 20:
            return "Strong Growth"

        if value >= 10:
            return "Healthy Growth"

        if value >= 0:
            return "Stable"

        return "Declining"

    # ======================================================
    # Build Financial Summary
    # ======================================================

    def build(
        self,
        financial_df: pd.DataFrame
    ) -> Dict:

        if financial_df is None or financial_df.empty:
            return {}

        print("=" * 80)
        print("NORMALIZED FINANCIAL DATA")
        print("=" * 80)
        print(financial_df.columns.tolist())
        print(financial_df.tail())
        print("=" * 80)

        latest = financial_df.iloc[-1]

        summary = {}

        # ======================================================
        # Latest Financial Values
        # ======================================================

        revenue = latest.get("revenue")
        ebit = latest.get("ebit")
        net_income = latest.get("net_income")
        cfo = latest.get("cash_from_operations")
        capex = latest.get("capex")
        debt = latest.get("total_debt")
        equity = latest.get("total_equity")
        shares = latest.get("shares_outstanding")

        # ======================================================
        # Raw Metrics
        # ======================================================

        summary["Revenue"] = (
            revenue if pd.notna(revenue) else "Unavailable"
        )

        summary["EBIT"] = (
            ebit if pd.notna(ebit) else "Unavailable"
        )

        summary["Net Income"] = (
            net_income if pd.notna(net_income) else "Unavailable"
        )

        # ======================================================
        # Free Cash Flow
        # ======================================================

        if pd.notna(cfo) and pd.notna(capex):

            free_cash_flow = cfo - capex

            summary["Free Cash Flow"] = free_cash_flow

        else:

            free_cash_flow = None

            summary["Free Cash Flow"] = "Unavailable"

        # ======================================================
        # Operating Margin
        # ======================================================

        if (
            pd.notna(revenue)
            and pd.notna(ebit)
            and revenue != 0
        ):

            summary["Operating Margin"] = round(

                (ebit / revenue) * 100,

                2

            )

        else:

            summary["Operating Margin"] = "Unavailable"

        # ======================================================
        # Net Margin
        # ======================================================

        if (
            pd.notna(revenue)
            and pd.notna(net_income)
            and revenue != 0
        ):

            summary["Net Margin"] = round(

                (net_income / revenue) * 100,

                2

            )

        else:

            summary["Net Margin"] = "Unavailable"

        # ======================================================
        # EPS
        # ======================================================

        if (
            pd.notna(net_income)
            and pd.notna(shares)
            and shares != 0
        ):

            summary["EPS"] = round(

                net_income / shares,

                2

            )

        else:

            summary["EPS"] = "Unavailable"

        # ======================================================
        # ROE
        # ======================================================

        if (
            pd.notna(net_income)
            and pd.notna(equity)
            and equity != 0
        ):

            summary["ROE"] = round(

                (net_income / equity) * 100,

                2

            )

        else:

            summary["ROE"] = "Unavailable"

        # ======================================================
        # Debt to Equity
        # ======================================================

        if (
            pd.notna(debt)
            and pd.notna(equity)
            and equity != 0
        ):

            summary["Debt to Equity"] = round(

                debt / equity,

                2

            )

        else:

            summary["Debt to Equity"] = "Unavailable"

        # ======================================================
        # Revenue Growth
        # ======================================================

        if len(financial_df) >= 2:

            previous = financial_df.iloc[-2]

            previous_revenue = previous.get("revenue")

            if (
                pd.notna(previous_revenue)
                and previous_revenue != 0
                and pd.notna(revenue)
            ):

                summary["Revenue Growth (%)"] = round(

                    ((revenue - previous_revenue) / previous_revenue) * 100,

                    2

                )

            else:

                summary["Revenue Growth (%)"] = "Unavailable"

        else:

            summary["Revenue Growth (%)"] = "Unavailable"

        # ======================================================
        # EBIT Growth
        # ======================================================

        if len(financial_df) >= 2:

            previous_ebit = financial_df.iloc[-2].get("ebit")

            if (
                pd.notna(previous_ebit)
                and previous_ebit != 0
                and pd.notna(ebit)
            ):

                summary["EBIT Growth (%)"] = round(

                    ((ebit - previous_ebit) / previous_ebit) * 100,

                    2

                )

            else:

                summary["EBIT Growth (%)"] = "Unavailable"

        else:

            summary["EBIT Growth (%)"] = "Unavailable"

        # ======================================================
        # Net Income Growth
        # ======================================================

        if len(financial_df) >= 2:

            previous_income = financial_df.iloc[-2].get("net_income")

            if (
                pd.notna(previous_income)
                and previous_income != 0
                and pd.notna(net_income)
            ):

                summary["Net Income Growth (%)"] = round(

                    ((net_income - previous_income) / previous_income) * 100,

                    2

                )

            else:

                summary["Net Income Growth (%)"] = "Unavailable"

        else:

            summary["Net Income Growth (%)"] = "Unavailable"

        # ======================================================
        # Free Cash Flow Growth
        # ======================================================

        if len(financial_df) >= 2:

            previous_cfo = financial_df.iloc[-2].get("cash_from_operations")
            previous_capex = financial_df.iloc[-2].get("capex")

            if (
                pd.notna(previous_cfo)
                and pd.notna(previous_capex)
                and free_cash_flow is not None
            ):

                previous_fcf = previous_cfo - previous_capex

                if previous_fcf != 0:

                    summary["FCF Growth (%)"] = round(

                        (
                            (free_cash_flow - previous_fcf)
                            / previous_fcf
                        ) * 100,

                        2

                    )

                else:

                    summary["FCF Growth (%)"] = "Unavailable"

            else:

                summary["FCF Growth (%)"] = "Unavailable"

        else:

            summary["FCF Growth (%)"] = "Unavailable"

        # ======================================================
        # Revenue CAGR
        # ======================================================

        revenues = financial_df["revenue"].dropna()

        if len(revenues) >= 3:

            start = revenues.iloc[0]
            end = revenues.iloc[-1]

            years = len(revenues) - 1

            cagr = (

                (end / start) ** (1 / years)

                - 1

            ) * 100

            summary["Revenue CAGR (%)"] = round(cagr, 2)

        else:

            summary["Revenue CAGR (%)"] = "Unavailable"

        # ======================================================
        # Trend Labels
        # ======================================================

        summary["Revenue Trend"] = self._trend(
            summary["Revenue Growth (%)"]
        )

        summary["EBIT Trend"] = self._trend(
            summary["EBIT Growth (%)"]
        )

        summary["Net Income Trend"] = self._trend(
            summary["Net Income Growth (%)"]
        )

        summary["FCF Trend"] = self._trend(
            summary["FCF Growth (%)"]
        )

        print("\nFINANCIAL SUMMARY")
        print(summary)
        print("=" * 80)

        return summary