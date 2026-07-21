"""
relative_valuation.py

Cross-checks the DCF's Buy/Hold/Sell call against a simple relative
valuation signal. A DCF's conclusion is only as good as its growth
and discount-rate assumptions -- a second, independent lens catches
cases where the DCF's assumptions, not the company's fundamentals,
are driving a Sell.

This compares the company against ITS OWN 3-5 year trading history
rather than sector/industry peers: this pipeline has no peer-multiple
data source (yfinance doesn't expose a reliable "peer group" or
industry-median-multiple API), so a peer comparison would mean either
a hardcoded, unmaintained peer list or an unreliable scrape. A
company's own trading history requires no new data source -- it
reuses normalized_financials and historical_prices, both already
fetched by MarketDataTool -- and answers a real, well-defined
question ("is this cheap or expensive relative to where THIS stock
has historically traded"), just a different question than "vs peers."
"""

import pandas as pd


class RelativeValuationEngine:

    # Current multiple within this band of the historical average is
    # treated as "in line" rather than cheap/expensive, so small,
    # noisy differences aren't over-interpreted as a signal.
    NEUTRAL_BAND_PCT = 10.0

    def __init__(self, financial_df, historical_prices, market_cap, current_price):
        self.financial_df = financial_df
        self.historical_prices = self._tz_naive(historical_prices)
        self.market_cap = market_cap
        self.current_price = current_price

    @staticmethod
    def _tz_naive(prices):
        if prices is None or prices.empty:
            return prices
        if prices.index.tz is not None:
            prices = prices.copy()
            prices.index = prices.index.tz_localize(None)
        return prices

    def _year_end_price(self, date):
        """Nearest available close on/before `date`."""
        if self.historical_prices is None or self.historical_prices.empty:
            return None
        prices = self.historical_prices["Close"]
        prices = prices[prices.index <= date]
        if prices.empty:
            return None
        return prices.iloc[-1]

    def _yearly_ev_ebitda(self):
        df = self.financial_df
        required = ["ebit", "depreciation", "total_debt", "cash", "shares_outstanding"]
        if not all(c in df.columns for c in required):
            return pd.Series(dtype=float)

        rows = {}
        for date, row in df.iterrows():
            ebit, depreciation, debt, cash, shares = (
                row.get("ebit"), row.get("depreciation"),
                row.get("total_debt"), row.get("cash"), row.get("shares_outstanding"),
            )
            if any(pd.isna(v) for v in (ebit, depreciation, debt, cash, shares)):
                continue

            ebitda = ebit + depreciation
            if ebitda <= 0:
                continue

            price = self._year_end_price(date)
            if price is None:
                continue

            approx_market_cap = price * shares
            ev = approx_market_cap + debt - cash
            rows[date] = ev / ebitda

        return pd.Series(rows).sort_index()

    def _yearly_ev_revenue(self):
        """
        Fallback multiple for companies with negative/unusable EBITDA
        (confirmed in Phase 2 backtesting: unprofitable growth
        companies -- RIVN, LCID, SNOW, U, AI, UPST, AFRM, RBLX, DKNG,
        PATH -- got no relative-valuation signal at all, since
        _yearly_ev_ebitda's `ebitda <= 0` filter drops every one of
        their fiscal years, and evaluate() also requires positive
        CURRENT EBITDA before returning anything). Revenue is
        virtually never negative, so EV/Revenue keeps a signal
        available for exactly the companies EV/EBITDA structurally
        can't cover -- the same "vs. its own trading history" logic,
        just on a metric that doesn't require profitability.
        """
        df = self.financial_df
        required = ["revenue", "total_debt", "cash", "shares_outstanding"]
        if not all(c in df.columns for c in required):
            return pd.Series(dtype=float)

        rows = {}
        for date, row in df.iterrows():
            revenue, debt, cash, shares = (
                row.get("revenue"), row.get("total_debt"),
                row.get("cash"), row.get("shares_outstanding"),
            )
            if any(pd.isna(v) for v in (revenue, debt, cash, shares)):
                continue

            if revenue <= 0:
                continue

            price = self._year_end_price(date)
            if price is None:
                continue

            approx_market_cap = price * shares
            ev = approx_market_cap + debt - cash
            rows[date] = ev / revenue

        return pd.Series(rows).sort_index()

    def _yearly_pfcf(self):
        df = self.financial_df
        required = ["cash_from_operations", "capex", "shares_outstanding"]
        if not all(c in df.columns for c in required):
            return pd.Series(dtype=float)

        rows = {}
        for date, row in df.iterrows():
            cfo, capex, shares = (
                row.get("cash_from_operations"), row.get("capex"), row.get("shares_outstanding"),
            )
            if any(pd.isna(v) for v in (cfo, capex, shares)):
                continue

            fcf = cfo - capex
            if fcf <= 0:
                continue

            price = self._year_end_price(date)
            if price is None:
                continue

            rows[date] = price / (fcf / shares)

        return pd.Series(rows).sort_index()

    def _signal_from_pct(self, pct_vs_history):
        if pct_vs_history <= -self.NEUTRAL_BAND_PCT:
            return "cheap"
        elif pct_vs_history >= self.NEUTRAL_BAND_PCT:
            return "expensive"
        return "in-line"

    def _evaluate_ev_ebitda(self):
        """Primary metric. Returns None if there isn't enough data to
        form a signal (fewer than 2 fiscal years with clean data, no
        market cap, or non-positive current EBITDA) -- most commonly
        that last case, which is exactly when _evaluate_ev_revenue
        below is tried instead."""
        ev_ebitda_series = self._yearly_ev_ebitda()
        if len(ev_ebitda_series) < 2 or self.market_cap is None:
            return None

        latest = self.financial_df.iloc[-1]
        ebit_current = latest.get("ebit")
        depreciation_current = latest.get("depreciation")
        debt_current = latest.get("total_debt")
        cash_current = latest.get("cash")

        if any(pd.isna(v) for v in (ebit_current, depreciation_current)):
            return None

        ebitda_current = ebit_current + depreciation_current
        if ebitda_current <= 0:
            return None

        ev_current = self.market_cap + (debt_current or 0) - (cash_current or 0)
        ev_ebitda_current = ev_current / ebitda_current

        # Historical average excludes the most recent fiscal year --
        # that row uses a year-end price approximation, while "current"
        # already uses today's exact market cap. Mixing them would
        # double-count roughly the same period two different ways.
        historical_series = ev_ebitda_series.iloc[:-1]
        historical_avg = historical_series.mean()

        if pd.isna(historical_avg) or historical_avg == 0:
            return None

        pct_vs_history = (ev_ebitda_current / historical_avg - 1) * 100

        return {
            "method": "Own trading history (no peer/sector multiple data "
                       "source is wired into this pipeline)",
            "metric": "EV/EBITDA",
            "current_ev_ebitda": round(ev_ebitda_current, 2),
            "historical_avg_ev_ebitda": round(historical_avg, 2),
            "years_used": len(historical_series),
            "vs_history_pct": round(pct_vs_history, 2),
            "signal": self._signal_from_pct(pct_vs_history),
        }

    def _evaluate_ev_revenue(self):
        """Fallback metric, tried only when _evaluate_ev_ebitda returns
        None -- see _yearly_ev_revenue for why this exists. Same shape
        and same neutral-band logic as the EBITDA path, just on
        revenue instead."""
        ev_revenue_series = self._yearly_ev_revenue()
        if len(ev_revenue_series) < 2 or self.market_cap is None:
            return None

        latest = self.financial_df.iloc[-1]
        revenue_current = latest.get("revenue")
        debt_current = latest.get("total_debt")
        cash_current = latest.get("cash")

        if pd.isna(revenue_current) or revenue_current <= 0:
            return None

        ev_current = self.market_cap + (debt_current or 0) - (cash_current or 0)
        ev_revenue_current = ev_current / revenue_current

        historical_series = ev_revenue_series.iloc[:-1]
        historical_avg = historical_series.mean()

        if pd.isna(historical_avg) or historical_avg == 0:
            return None

        pct_vs_history = (ev_revenue_current / historical_avg - 1) * 100

        return {
            "method": (
                "Own trading history (EV/Revenue fallback -- EBITDA is negative "
                "or unusable for this company, so EV/EBITDA can't form a signal; "
                "no peer/sector multiple data source is wired into this pipeline)"
            ),
            "metric": "EV/Revenue",
            # Reuses the EV/EBITDA result's field names deliberately --
            # every caller (report_data_builder.py's fallback-recommendation
            # basis text, pdf_report_builder.py's "Current Multiple" row)
            # already reads these generically as "the current multiple" and
            # labels which metric it is separately via "metric" above; a
            # second pair of metric-specific field names would just make
            # every caller branch on which metric was used to display a
            # number that's conceptually the same thing either way.
            "current_ev_ebitda": round(ev_revenue_current, 2),
            "historical_avg_ev_ebitda": round(historical_avg, 2),
            "years_used": len(historical_series),
            "vs_history_pct": round(pct_vs_history, 2),
            "signal": self._signal_from_pct(pct_vs_history),
        }

    def evaluate(self):
        """
        Returns None if there isn't enough data to form ANY signal.
        Tries EV/EBITDA first (_evaluate_ev_ebitda, the more standard
        profitability-based multiple); falls back to EV/Revenue
        (_evaluate_ev_revenue) when EBITDA is negative/unusable, which
        is common for unprofitable growth companies that would
        otherwise get no relative-valuation signal at all -- confirmed
        in Phase 2 backtesting (RIVN, LCID, SNOW, U, AI, UPST, AFRM,
        RBLX, DKNG, PATH all fell into this gap). A P/FCF cross-check
        is attached when available, independent of which primary
        metric was used.
        """
        result = self._evaluate_ev_ebitda() or self._evaluate_ev_revenue()
        if result is None:
            return None

        latest = self.financial_df.iloc[-1]
        pfcf_series = self._yearly_pfcf()
        cfo_current, capex_current, shares_current = (
            latest.get("cash_from_operations"), latest.get("capex"), latest.get("shares_outstanding"),
        )
        if (
            len(pfcf_series) >= 2
            and self.current_price
            and not any(pd.isna(v) for v in (cfo_current, capex_current, shares_current))
        ):
            fcf_current = cfo_current - capex_current
            if fcf_current > 0:
                pfcf_current = self.current_price / (fcf_current / shares_current)
                pfcf_historical_avg = pfcf_series.iloc[:-1].mean()
                if not pd.isna(pfcf_historical_avg):
                    result["current_pfcf"] = round(pfcf_current, 2)
                    result["historical_avg_pfcf"] = round(pfcf_historical_avg, 2)

        return result
