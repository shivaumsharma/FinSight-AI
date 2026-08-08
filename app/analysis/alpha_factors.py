"""
alpha_factors.py

Computes a transparent "Alpha Factors" scorecard -- ~30 named financial/
valuation/quality/market/risk/sentiment/macro signals -- for display
alongside the DCF/relative-valuation/Monte Carlo/ML-classifier sections of
a report. Deliberately NOT wired into report_data_builder.py's
recommendation composite (DCF_WEIGHT/RELATIVE_WEIGHT/_composite_score) --
same treatment as the ML valuation classifier (see its own docstring):
shown for transparency only, no track record yet informing Buy/Hold/Sell.

Every factor degrades independently -- a missing/NaN input blanks out only
that one factor (None), never the whole scorecard. Mirrors
FinancialAnalysisBuilder's own "Unavailable" idiom and
derive_recommendation's stated "NaN is treated like missing, not like a
real number" philosophy.

Three factors (Relative Strength vs Index, Sector Relative Performance,
Interest Rate Sensitivity) compare the company against a USD-denominated
benchmark and are gated off entirely for NSE (.NS) tickers -- comparing a
rupee-denominated stock's returns against the S&P 500 / a US sector ETF /
US Treasury yields doesn't produce a meaningful signal (no FX adjustment,
wrong market's cycle), so this states the real coverage gap explicitly
instead of computing a misleading number -- same honesty-over-completeness
precedent as report_data_builder.py's own filing_evidence_note.
"""

import math
from typing import Any, Dict, Optional

import pandas as pd

MOMENTUM_WINDOW_6M = 126
MOMENTUM_WINDOW_12M = 252


def _clean(value):
    """None/NaN -> None, else unchanged -- so a NaN read off a DataFrame
    is never mistaken for a real number anywhere below."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


class AlphaFactorsEngine:

    # Matches RelativeValuationEngine's own neutral band -- small,
    # noisy differences from the historical average aren't
    # over-interpreted as a cheap/expensive signal.
    NEUTRAL_BAND_PCT = 10.0

    def __init__(
        self,
        normalized_financials: Optional[pd.DataFrame],
        historical_prices: Optional[pd.DataFrame],
        beta: Optional[float],
        company_info: Dict[str, Any],
        financial_summary: Dict[str, Any],
        sentiment_summary: Dict[str, Any],
        news_sentiment_summary: Dict[str, Any],
        is_non_us_listing: bool,
        benchmark_history: Optional[pd.DataFrame] = None,
        sector_history: Optional[pd.DataFrame] = None,
        rate_proxy_history: Optional[pd.DataFrame] = None,
    ):
        self.df = normalized_financials
        self.prices = self._tz_naive(historical_prices)
        self.beta = beta
        self.company_info = company_info or {}
        self.financial_summary = financial_summary or {}
        self.sentiment_summary = sentiment_summary or {}
        self.news_sentiment_summary = news_sentiment_summary or {}
        self.is_non_us_listing = is_non_us_listing
        self.benchmark_history = self._tz_naive(benchmark_history)
        self.sector_history = self._tz_naive(sector_history)
        self.rate_proxy_history = self._tz_naive(rate_proxy_history)

    @staticmethod
    def _tz_naive(prices):
        if prices is None or prices.empty:
            return prices
        if prices.index.tz is not None:
            prices = prices.copy()
            prices.index = prices.index.tz_localize(None)
        return prices

    def _year_end_price(self, date):
        """Nearest available close on/before `date` -- same lookup as
        RelativeValuationEngine._year_end_price, reimplemented locally
        rather than shared (this codebase's own established precedent
        for small helpers -- see scripts/phase2_backtest.py's own
        independent _tz_naive)."""
        if self.prices is None or self.prices.empty or "Close" not in self.prices.columns:
            return None
        prices = self.prices["Close"]
        prices = prices[prices.index <= date]
        if prices.empty:
            return None
        return prices.iloc[-1]

    def _xref(self, key: str):
        """Read an already-computed value out of financial_summary,
        treating its "Unavailable" string sentinel the same as missing."""
        value = self.financial_summary.get(key)
        if value == "Unavailable":
            return None
        return _clean(value)

    def _has_columns(self, *cols) -> bool:
        return self.df is not None and all(c in self.df.columns for c in cols)

    def _latest(self, *cols):
        """Latest fiscal year's value(s) for the given column(s) -- None
        for any column missing from the DataFrame or NaN in the latest
        row. Returns a scalar for one column, a tuple for several."""
        if not self._has_columns(*cols) or self.df.empty:
            return None if len(cols) == 1 else (None,) * len(cols)
        latest = self.df.iloc[-1]
        values = [_clean(latest.get(c)) for c in cols]
        return values[0] if len(cols) == 1 else tuple(values)

    # ------------------------------------------------------------------
    # Financial
    # ------------------------------------------------------------------

    def _financial_factors(self) -> Dict[str, Any]:
        factors: Dict[str, Any] = {
            "Revenue CAGR (%)": self._xref("Revenue CAGR (%)"),
            "Revenue Growth YoY (%)": self._xref("Revenue Growth (%)"),
            "EBIT Growth YoY (%)": self._xref("EBIT Growth (%)"),
            "Net Income Growth YoY (%)": self._xref("Net Income Growth (%)"),
            "FCF Growth YoY (%)": self._xref("FCF Growth (%)"),
            "Operating Margin (%)": self._xref("Operating Margin"),
            "Net Margin (%)": self._xref("Net Margin"),
        }

        revenue, ebit, depreciation = self._latest("revenue", "ebit", "depreciation")
        if revenue and ebit is not None and depreciation is not None:
            factors["EBITDA Margin (%)"] = round((ebit + depreciation) / revenue * 100, 2)
        else:
            factors["EBITDA Margin (%)"] = None

        revenue_val, fcf_val = self._xref("Revenue"), self._xref("Free Cash Flow")
        if revenue_val and fcf_val is not None:
            factors["FCF Margin (%)"] = round(fcf_val / revenue_val * 100, 2)
        else:
            factors["FCF Margin (%)"] = None

        return factors

    # ------------------------------------------------------------------
    # Quality
    # ------------------------------------------------------------------

    def _quality_factors(self) -> Dict[str, Any]:
        factors: Dict[str, Any] = {
            "Return on Equity (%)": self._xref("ROE"),
            "Debt to Equity": self._xref("Debt to Equity"),
        }

        # Each column fetched on its own single-column call, deliberately
        # NOT batched together -- _latest() requires every column in one
        # call to be present, so batching e.g. ebit with total_assets
        # would make Interest Coverage Ratio (needs only ebit) wrongly
        # degrade to None whenever total_assets alone is missing.
        total_assets = self._latest("total_assets")
        net_income = self._latest("net_income")
        ebit = self._latest("ebit")
        current_liabilities = self._latest("current_liabilities")
        current_assets = self._latest("current_assets")
        interest_expense = self._latest("interest_expense")

        factors["Return on Assets (%)"] = (
            round(net_income / total_assets * 100, 2)
            if total_assets and net_income is not None
            else None
        )

        capital_employed = (
            total_assets - current_liabilities
            if total_assets is not None and current_liabilities is not None
            else None
        )
        factors["Return on Capital Employed (%)"] = (
            round(ebit / capital_employed * 100, 2)
            if ebit is not None and capital_employed
            else None
        )

        factors["Current Ratio"] = (
            round(current_assets / current_liabilities, 2)
            if current_assets is not None and current_liabilities
            else None
        )

        factors["Interest Coverage Ratio"] = (
            round(ebit / abs(interest_expense), 2)
            if ebit is not None and interest_expense
            else None
        )

        factors["Piotroski F-Score (0-9, proxy)"] = self._piotroski_f_score()

        return factors

    # Standard 9-test Piotroski composite, with two disclosed
    # substitutions where this codebase's 15-column normalized schema
    # lacks the textbook input: total-debt-to-assets stands in for the
    # long-term-debt ratio (test 5, no clean current/non-current debt
    # split is available), and Δ Operating Margin stands in for Δ Gross
    # Margin (test 8, no COGS/gross-profit column exists). All-or-
    # nothing: any missing/NaN input, or fewer than 2 fiscal years,
    # returns None rather than a partial score with an unknown test
    # silently scored as a fail -- the 0-9 scale assumes all 9 are real
    # reads. Structurally None for banks/financials (missing ebit/
    # current_assets/current_liabilities), the same way DCF already is.
    _PIOTROSKI_COLUMNS = (
        "net_income", "total_assets", "cash_from_operations",
        "total_debt", "current_assets", "current_liabilities",
        "shares_outstanding", "ebit", "revenue",
    )

    def _piotroski_f_score(self) -> Optional[int]:
        if not self._has_columns(*self._PIOTROSKI_COLUMNS) or len(self.df) < 2:
            return None

        curr_row, prior_row = self.df.iloc[-1], self.df.iloc[-2]
        curr = {c: _clean(curr_row.get(c)) for c in self._PIOTROSKI_COLUMNS}
        prior = {c: _clean(prior_row.get(c)) for c in self._PIOTROSKI_COLUMNS}
        if any(v is None for v in curr.values()) or any(v is None for v in prior.values()):
            return None
        if curr["total_assets"] == 0 or prior["total_assets"] == 0:
            return None
        if curr["current_liabilities"] == 0 or prior["current_liabilities"] == 0:
            return None
        if curr["revenue"] == 0 or prior["revenue"] == 0:
            return None

        curr_roa = curr["net_income"] / curr["total_assets"]
        prior_roa = prior["net_income"] / prior["total_assets"]
        curr_leverage = curr["total_debt"] / curr["total_assets"]
        prior_leverage = prior["total_debt"] / prior["total_assets"]
        curr_current_ratio = curr["current_assets"] / curr["current_liabilities"]
        prior_current_ratio = prior["current_assets"] / prior["current_liabilities"]
        curr_op_margin = curr["ebit"] / curr["revenue"]
        prior_op_margin = prior["ebit"] / prior["revenue"]
        curr_turnover = curr["revenue"] / curr["total_assets"]
        prior_turnover = prior["revenue"] / prior["total_assets"]

        tests = [
            curr_roa > 0,
            curr["cash_from_operations"] > 0,
            curr_roa > prior_roa,
            curr["cash_from_operations"] > curr["net_income"],
            curr_leverage < prior_leverage,
            curr_current_ratio > prior_current_ratio,
            curr["shares_outstanding"] <= prior["shares_outstanding"],
            curr_op_margin > prior_op_margin,
            curr_turnover > prior_turnover,
        ]
        # sum() of numpy.bool_ comparisons (curr/prior come from a
        # pandas Series, so these are numpy scalars, not Python floats)
        # returns numpy.int64, not a plain Python int -- confirmed via a
        # real live run (not caught by unit tests, which build their
        # DataFrames from Python float literals): FastAPI's default JSON
        # encoder serializes numpy.float64 fine (it subclasses float)
        # but raises "Object of type int64 is not JSON serializable" on
        # numpy.int64, which does NOT subclass Python's int.
        return int(sum(tests))

    # Original (1968) public-company Altman Z-Score, market value of
    # equity -- not the private-company Z'-Score variant. total_liabilities
    # is derived as total_assets - total_equity rather than adding a third
    # balance-sheet mapping. All-or-nothing degradation, same reasoning as
    # Piotroski. A known, accepted academic limitation: the model was
    # calibrated on manufacturing companies; applying it uniformly to
    # services/tech (where it still computes) is a standard limitation of
    # using this exact model, not something this module can fix.
    _ALTMAN_COLUMNS = (
        "current_assets", "current_liabilities", "total_assets",
        "retained_earnings", "ebit", "total_equity", "revenue",
    )

    def _altman_z_score(self) -> Optional[float]:
        if not self._has_columns(*self._ALTMAN_COLUMNS) or self.df.empty:
            return None
        latest = self.df.iloc[-1]
        vals = {c: _clean(latest.get(c)) for c in self._ALTMAN_COLUMNS}
        market_cap = _clean(self.company_info.get("market_cap"))
        if any(v is None for v in vals.values()) or market_cap is None:
            return None
        total_assets = vals["total_assets"]
        if not total_assets:
            return None
        total_liabilities = total_assets - vals["total_equity"]
        if not total_liabilities:
            return None

        z = (
            1.2 * (vals["current_assets"] - vals["current_liabilities"]) / total_assets
            + 1.4 * vals["retained_earnings"] / total_assets
            + 3.3 * vals["ebit"] / total_assets
            + 0.6 * market_cap / total_liabilities
            + 1.0 * vals["revenue"] / total_assets
        )
        return round(z, 2)

    @staticmethod
    def _altman_zone(z: Optional[float]) -> Optional[str]:
        if z is None:
            return None
        if z > 2.99:
            return "Safe Zone"
        if z >= 1.81:
            return "Grey Zone"
        return "Distress Zone"

    # ------------------------------------------------------------------
    # Valuation -- P/E and P/B vs the company's OWN 3-5yr trading
    # history, not vs sector/peers. Mirrors RelativeValuationEngine's
    # own deliberate choice (no reliable peer/sector multiple data
    # source is wired into this pipeline -- see that module's own
    # docstring) rather than skipping these two factors or building an
    # unreliable peer comparison. EV/EBITDA, P/FCF, and Monte Carlo's
    # prob_undervalued are deliberately excluded here: already computed
    # and already displayed elsewhere, so including them would be
    # redundant, not additive.
    # ------------------------------------------------------------------

    def _yearly_pe(self) -> pd.Series:
        if not self._has_columns("net_income", "shares_outstanding"):
            return pd.Series(dtype=float)
        rows = {}
        for date, row in self.df.iterrows():
            net_income, shares = _clean(row.get("net_income")), _clean(row.get("shares_outstanding"))
            if net_income is None or not shares:
                continue
            eps = net_income / shares
            if eps <= 0:
                continue
            price = self._year_end_price(date)
            if price is None:
                continue
            rows[date] = price / eps
        return pd.Series(rows).sort_index()

    def _yearly_pb(self) -> pd.Series:
        if not self._has_columns("total_equity", "shares_outstanding"):
            return pd.Series(dtype=float)
        rows = {}
        for date, row in self.df.iterrows():
            equity, shares = _clean(row.get("total_equity")), _clean(row.get("shares_outstanding"))
            if not equity or equity <= 0 or not shares:
                continue
            bvps = equity / shares
            price = self._year_end_price(date)
            if price is None:
                continue
            rows[date] = price / bvps
        return pd.Series(rows).sort_index()

    def _signal_from_pct(self, pct: float) -> str:
        if pct <= -self.NEUTRAL_BAND_PCT:
            return "cheap"
        if pct >= self.NEUTRAL_BAND_PCT:
            return "expensive"
        return "in-line"

    def _vs_history(
        self, series: pd.Series, current_price, current_denominator, label: str
    ) -> Optional[Dict[str, Any]]:
        """Shared 'current multiple vs own trading history' shape, same
        rules as RelativeValuationEngine._evaluate_ev_ebitda: >=2
        historical years required, latest year excluded from the
        historical average (it mixes a year-end price approximation
        with a same-period comparison "current" already uses exactly)."""
        if len(series) < 2 or not current_price or not current_denominator or current_denominator <= 0:
            return None
        current_multiple = current_price / current_denominator
        historical_avg = series.iloc[:-1].mean()
        if pd.isna(historical_avg) or historical_avg == 0:
            return None
        pct_vs_history = (current_multiple / historical_avg - 1) * 100
        return {
            "metric": label,
            "current": round(current_multiple, 2),
            "historical_avg": round(historical_avg, 2),
            "years_used": len(series) - 1,
            "vs_history_pct": round(pct_vs_history, 2),
            "signal": self._signal_from_pct(pct_vs_history),
        }

    def _valuation_factors(self) -> Dict[str, Any]:
        current_price = _clean(self.company_info.get("current_price"))

        eps = self._xref("EPS")
        pe_result = self._vs_history(self._yearly_pe(), current_price, eps, "P/E") if eps and eps > 0 else None

        total_equity, shares = self._latest("total_equity", "shares_outstanding")
        bvps = total_equity / shares if total_equity and total_equity > 0 and shares else None
        pb_result = self._vs_history(self._yearly_pb(), current_price, bvps, "P/B") if bvps else None

        return {
            "P/E vs Own History": pe_result,
            "P/B vs Own History": pb_result,
        }

    # ------------------------------------------------------------------
    # Market -- pure historical_prices["Close"], zero new fetch for the
    # first four; Relative Strength/Sector Performance need the
    # (optional, benchmark-supplied) index/sector series.
    # ------------------------------------------------------------------

    def _return_over_window(self, prices: Optional[pd.DataFrame], window: int) -> Optional[float]:
        if prices is None or prices.empty or "Close" not in prices.columns:
            return None
        close = prices["Close"].dropna()
        if len(close) < window:
            return None
        past = close.iloc[-window]
        if not past:
            return None
        return (close.iloc[-1] / past - 1) * 100

    def _market_factors(self) -> Dict[str, Any]:
        factors: Dict[str, Any] = {
            "Price vs 52-Week High (%)": None,
            "Price vs 52-Week Low (%)": None,
            "6-Month Price Momentum (%)": None,
            "12-Month Price Momentum (%)": None,
            "Relative Strength vs Index (%)": None,
            "Sector Relative Performance (%)": None,
        }
        if self.prices is None or self.prices.empty or "Close" not in self.prices.columns:
            return factors

        close = self.prices["Close"].dropna()
        current = close.iloc[-1] if not close.empty else None

        if current is not None and len(close) >= MOMENTUM_WINDOW_12M:
            window = close.tail(MOMENTUM_WINDOW_12M)
            high, low = window.max(), window.min()
            if high:
                factors["Price vs 52-Week High (%)"] = round((current / high - 1) * 100, 2)
            if low:
                factors["Price vs 52-Week Low (%)"] = round((current / low - 1) * 100, 2)

        twelve_month = self._return_over_window(self.prices, MOMENTUM_WINDOW_12M)
        if twelve_month is not None:
            factors["12-Month Price Momentum (%)"] = round(twelve_month, 2)

        six_month = self._return_over_window(self.prices, MOMENTUM_WINDOW_6M)
        if six_month is not None:
            factors["6-Month Price Momentum (%)"] = round(six_month, 2)

        if self.is_non_us_listing:
            factors["_market_note"] = (
                "Relative Strength vs Index and Sector Relative Performance are not "
                "computed for NSE-listed tickers: no Nifty/Sensex-equivalent benchmark "
                "or sector-index data source is wired into this pipeline yet, and "
                "comparing against a US benchmark would mix currencies/markets in a "
                "way that isn't a meaningful signal."
            )
        else:
            benchmark_6m = self._return_over_window(self.benchmark_history, MOMENTUM_WINDOW_6M)
            if six_month is not None and benchmark_6m is not None:
                factors["Relative Strength vs Index (%)"] = round(six_month - benchmark_6m, 2)

            sector_6m = self._return_over_window(self.sector_history, MOMENTUM_WINDOW_6M)
            if six_month is not None and sector_6m is not None:
                factors["Sector Relative Performance (%)"] = round(six_month - sector_6m, 2)

        return factors

    # ------------------------------------------------------------------
    # Risk
    # ------------------------------------------------------------------

    def _annualized_volatility(self) -> Optional[float]:
        if self.prices is None or self.prices.empty or "Close" not in self.prices.columns:
            return None
        close = self.prices["Close"].dropna()
        if len(close) < MOMENTUM_WINDOW_12M:
            return None
        returns = close.tail(MOMENTUM_WINDOW_12M).pct_change().dropna()
        if returns.empty:
            return None
        return round(returns.std() * math.sqrt(252) * 100, 2)

    def _risk_factors(self) -> Dict[str, Any]:
        z_score = self._altman_z_score()
        return {
            "Beta": round(self.beta, 2) if self.beta is not None else None,
            "Annualized Volatility (%)": self._annualized_volatility(),
            "Altman Z-Score": z_score,
            "Altman Z-Score Zone": self._altman_zone(z_score),
        }

    # ------------------------------------------------------------------
    # Sentiment -- pure cross-reference into a unified scorecard, not
    # recomputation. Both values already render as ToneTiles at the top
    # of the report; this surfaces them again alongside every other
    # factor for one-screen comparability.
    # ------------------------------------------------------------------

    def _sentiment_factors(self) -> Dict[str, Any]:
        return {
            "Management/SEC Filing Sentiment": self.sentiment_summary.get("Overall Sentiment"),
            "News/Media Sentiment": self.news_sentiment_summary.get("Overall Sentiment"),
        }

    # ------------------------------------------------------------------
    # Macro -- deliberately thin. Everything else genuinely "macro"
    # (yield curve, GDP, PMI, commodities, FII/DII flows) needs a data
    # source this pipeline doesn't have, same category of gap already
    # accepted for FII/DII earlier.
    # ------------------------------------------------------------------

    def _interest_rate_sensitivity(self) -> Optional[Dict[str, Any]]:
        if self.is_non_us_listing:
            return None
        if self.prices is None or self.prices.empty or "Close" not in self.prices.columns:
            return None
        if (
            self.rate_proxy_history is None
            or self.rate_proxy_history.empty
            or "Close" not in self.rate_proxy_history.columns
        ):
            return None

        stock_returns = self.prices["Close"].dropna().pct_change().dropna()
        yield_changes = self.rate_proxy_history["Close"].dropna().diff().dropna()

        aligned = pd.concat([stock_returns, yield_changes], axis=1, join="inner")
        aligned.columns = ["stock_return", "yield_change"]
        aligned = aligned.tail(MOMENTUM_WINDOW_12M).dropna()
        if len(aligned) < 30:
            return None

        variance = aligned["yield_change"].var()
        if not variance or pd.isna(variance):
            return None
        sensitivity = aligned["stock_return"].cov(aligned["yield_change"]) / variance

        return {
            "coefficient": round(sensitivity, 4),
            "definition": (
                "Regression coefficient of daily stock returns against daily "
                "percentage-point changes in the 10-year Treasury yield (^TNX) over "
                "the trailing 12 months. Units: % stock return per 1-percentage-point "
                "yield move -- a different, less-standardized quantity than beta "
                "(which compares two return series), not directly comparable to it."
            ),
        }

    def _macro_factors(self) -> Dict[str, Any]:
        result = self._interest_rate_sensitivity()
        factors: Dict[str, Any] = {
            "Interest Rate Sensitivity": result["coefficient"] if result else None,
        }
        if result:
            factors["_interest_rate_sensitivity_definition"] = result["definition"]
        elif self.is_non_us_listing:
            factors["_macro_note"] = (
                "Interest Rate Sensitivity is not computed for NSE-listed tickers -- "
                "see the Market category note on cross-market benchmark gating."
            )
        return factors

    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, Dict[str, Any]]:
        return {
            "financial": self._financial_factors(),
            "quality": self._quality_factors(),
            "valuation": self._valuation_factors(),
            "market": self._market_factors(),
            "risk": self._risk_factors(),
            "sentiment": self._sentiment_factors(),
            "macro": self._macro_factors(),
        }
