import time
from datetime import date

import yfinance as yf
import pandas as pd

from app.core.cache import cache_get, cache_set, make_key
from app.core.retry import retry_on_transient_error

# TTL-only caching (see app/core/cache.py's module docstring for why
# this is a different strategy than the content-addressed caches in
# valuation_pipeline.py/narrative_builder.py): there's no per-request
# "input" to hash here besides the ticker itself -- the whole point is
# reusing a fetch across repeat requests within the same trading day.
# Deliberately scoped to ONLY the three statement fetches below, which
# genuinely only change quarterly -- get_company_info() (current_price,
# market_cap) and get_historical_prices() are NEVER cached this way,
# since serving a stale price on a live research tool would be a real
# correctness bug, not just a staleness inconvenience.
STATEMENT_CACHE_TTL_SECONDS = 12 * 3600


class TickerNotFoundError(Exception):
    """Raised when yfinance has no usable data for a ticker -- it
    doesn't exist, is delisted, or was mistyped. Caught centrally in
    streamlit_app.py to show a friendly message instead of a raw
    traceback from whichever statement fetch happens to hit an empty
    DataFrame first."""


class MarketDataUnavailableError(Exception):
    """Raised by market_data_tool.py when a data-provider fetch itself
    failed (even after retry_on_transient_error's retries exhausted) --
    e.g. a throttled/rate-limited yfinance response -- as opposed to
    TickerNotFoundError, which means the provider responded fine but
    genuinely had nothing for this ticker. The two must not be
    conflated: a throttled response about a perfectly valid ticker
    reported as "ticker not found" would be actively misleading. See
    MarketDataLoader.get_company_info's info_fetch_failed flag, which
    is the signal market_data_tool.py keys off to raise this instead
    of TickerNotFoundError."""


# Deliberately separate from get_company_info() below, not a thin
# wrapper around it -- get_company_info() makes two yfinance round
# trips (.info + .calendar) to fetch dozens of fields (business
# summary, employees, website...) a watchlist tile has no use for, and
# is intentionally never cached because a stale price in a *report*
# would be a correctness bug. A watchlist tile is a different case --
# normal quote-UI behavior every trading app caches briefly -- so this
# uses yfinance's lightweight fast_info accessor and a short-lived
# in-process cache, NOT app/core/cache.py (that's Redis-backed and
# silently no-ops without Redis; reusing it here would contradict this
# module's own "never cache price" policy for the wrong reason -- this
# genuinely is a different, cacheable case, not an exception to it).
_quote_cache: dict = {}
_QUOTE_CACHE_TTL_SECONDS = 45


def get_quote(ticker: str) -> dict:
    """Cheap current-price lookup for the Watchlist -- {"price",
    "change_pct", "previous_close", "currency"}. previous_close is
    exposed (already fetched internally to compute change_pct) so
    callers that need today's-dollar-move math (e.g. the Portfolio's
    Today's P&L) don't have to back-derive it from change_pct and risk
    float drift. currency (e.g. "USD", "INR") comes from fast_info at
    no extra network cost -- NOT a ticker-suffix guess -- and is what
    lets Portfolio/Watchlist show a rupee-priced holding with a rupee
    sign instead of always assuming dollars. Raises TickerNotFoundError
    if fast_info has no usable price (bad/delisted symbol), same
    exception the rest of this module already uses for that condition."""
    ticker = ticker.upper()

    cached = _quote_cache.get(ticker)
    if cached is not None and time.time() - cached[0] < _QUOTE_CACHE_TTL_SECONDS:
        return cached[1]

    try:
        fast_info = yf.Ticker(ticker).fast_info
        price = fast_info.last_price
        previous_close = fast_info.previous_close
        currency = fast_info.currency or "USD"
    except Exception as exc:
        # fast_info raises its own internal errors (e.g. KeyError deep
        # in yfinance's response parsing) for an invalid/delisted
        # ticker rather than returning empty data -- normalize all of
        # that to the one exception type this module already uses for
        # "no usable data for this ticker".
        raise TickerNotFoundError(f"No price data found for {ticker}") from exc
    if not price:
        raise TickerNotFoundError(f"No price data found for {ticker}")

    change_pct = ((price - previous_close) / previous_close * 100) if previous_close else None
    quote = {"price": price, "change_pct": change_pct, "previous_close": previous_close, "currency": currency}
    _quote_cache[ticker] = (time.time(), quote)
    return quote


# Only the one non-USD currency this app's actual company/index
# coverage needs (US via SEC, India via NSE -- see
# app/core/company_resolver.py's own scope) gets an FX ticker here.
# Yahoo quotes "INR=X" as INR-per-1-USD (the same convention Google/XE
# use for USD/INR), so converting a native amount TO USD divides by it.
_FX_TICKERS_BY_CURRENCY = {"INR": "INR=X"}


def get_usd_conversion_rate(currency: str):
    """How many USD equal 1 unit of `currency` -- 1.0 for USD (no
    conversion needed). Used by the Portfolio summary to make a mixed
    USD/INR total meaningful (each holding still displays in its own
    native currency; only the aggregate needs a common unit). Returns
    None if there's no known FX ticker for `currency` or its live quote
    fails -- callers must then exclude that holding from the aggregate
    rather than silently mixing units again."""
    if not currency or currency == "USD":
        return 1.0
    fx_ticker = _FX_TICKERS_BY_CURRENCY.get(currency)
    if fx_ticker is None:
        return None
    try:
        rate = get_quote(fx_ticker)["price"]
    except Exception:
        return None
    return (1.0 / rate) if rate else None


# Same lightweight-and-cached spirit as get_quote() above, for a
# different watchlist need: upcoming earnings + dividend dates, and the
# most recent split -- NOT routed through MarketDataLoader.get_company_info()
# (two round trips, dozens of unrelated fields, deliberately never
# cached). A longer TTL than quotes: these change at most a few times
# a year, not every 45 seconds.
_corporate_actions_cache: dict = {}
_CORPORATE_ACTIONS_CACHE_TTL_SECONDS = 3600


def get_corporate_actions(ticker: str) -> dict:
    """Cheap, never-raising lookup for the Watchlist -- {"next_earnings_date",
    "next_ex_dividend_date", "last_dividend_amount", "last_split"}, all
    None when yfinance has nothing to report. Mirrors
    MarketDataLoader.get_next_earnings_date's own never-break-the-caller
    contract for each individual field, so one missing data point
    (e.g. a company that's never split) doesn't blank out the rest."""
    ticker = ticker.upper()

    cached = _corporate_actions_cache.get(ticker)
    if cached is not None and time.time() - cached[0] < _CORPORATE_ACTIONS_CACHE_TTL_SECONDS:
        return cached[1]

    stock = yf.Ticker(ticker)

    next_earnings_date = None
    next_ex_dividend_date = None
    try:
        calendar = stock.calendar or {}
        earnings_dates = calendar.get("Earnings Date") or []
        upcoming_earnings = [d for d in earnings_dates if d >= date.today()]
        next_earnings_date = min(upcoming_earnings).isoformat() if upcoming_earnings else None

        ex_div = calendar.get("Ex-Dividend Date")
        if ex_div and ex_div >= date.today():
            next_ex_dividend_date = ex_div.isoformat()
    except Exception:
        pass

    last_dividend_amount = None
    try:
        dividends = stock.dividends
        if not dividends.empty:
            last_dividend_amount = float(dividends.iloc[-1])
    except Exception:
        pass

    last_split = None
    try:
        splits = stock.splits
        if not splits.empty:
            last_split = {
                "date": splits.index[-1].date().isoformat(),
                "ratio": float(splits.iloc[-1]),
            }
    except Exception:
        pass

    result = {
        "next_earnings_date": next_earnings_date,
        "next_ex_dividend_date": next_ex_dividend_date,
        "last_dividend_amount": last_dividend_amount,
        "last_split": last_split,
    }
    _corporate_actions_cache[ticker] = (time.time(), result)
    return result


_corporate_actions_history_cache: dict = {}
_CORPORATE_ACTIONS_HISTORY_CACHE_TTL_SECONDS = 6 * 3600


def get_corporate_actions_history(ticker: str) -> dict:
    """Full dividend/split history (not just the single most-recent
    each, unlike get_corporate_actions above) plus the same upcoming-
    earnings/ex-dividend lookahead -- powers the stock-detail page's
    Events tab. {"next_earnings_date", "next_ex_dividend_date",
    "dividends": [{"date", "amount"}, ...], "splits": [{"date",
    "ratio"}, ...]}, oldest first, both possibly empty (never None --
    a company that's simply never split has an empty list, not a
    missing field). No bonus/rights/buyback/ESOP history here --
    yfinance has no such concepts at all (see the approved stock-
    detail-page plan's "Explicitly out of scope" section); this is
    real dividend/split data only, not a guess at the missing kinds."""
    ticker = ticker.upper()

    cached = _corporate_actions_history_cache.get(ticker)
    if cached is not None and time.time() - cached[0] < _CORPORATE_ACTIONS_HISTORY_CACHE_TTL_SECONDS:
        return cached[1]

    stock = yf.Ticker(ticker)

    next_earnings_date = None
    next_ex_dividend_date = None
    try:
        calendar = stock.calendar or {}
        earnings_dates = calendar.get("Earnings Date") or []
        upcoming_earnings = [d for d in earnings_dates if d >= date.today()]
        next_earnings_date = min(upcoming_earnings).isoformat() if upcoming_earnings else None

        ex_div = calendar.get("Ex-Dividend Date")
        if ex_div and ex_div >= date.today():
            next_ex_dividend_date = ex_div.isoformat()
    except Exception:
        pass

    dividends = []
    try:
        series = stock.dividends
        if not series.empty:
            dividends = [
                {"date": idx.date().isoformat(), "amount": float(value)}
                for idx, value in series.items()
            ]
    except Exception:
        pass

    splits = []
    try:
        series = stock.splits
        if not series.empty:
            splits = [
                {"date": idx.date().isoformat(), "ratio": float(value)}
                for idx, value in series.items()
            ]
    except Exception:
        pass

    result = {
        "next_earnings_date": next_earnings_date,
        "next_ex_dividend_date": next_ex_dividend_date,
        "dividends": dividends,
        "splits": splits,
    }
    _corporate_actions_history_cache[ticker] = (time.time(), result)
    return result


# Same lightweight-and-cached spirit as get_quote()/get_corporate_actions()
# above, for AlphaFactorsEngine's market/risk/macro factors (Relative
# Strength vs Index, Sector Relative Performance, Interest Rate
# Sensitivity -- see app/analysis/alpha_factors.py). A 6-hour TTL, not
# 45s like get_quote(): this is 5 years of daily-close reference data,
# identical for every single research request on a given day, not a
# live intraday price -- 6h avoids re-fetching ^GSPC/^TNX/a sector ETF
# on every ticker's report while staying fresh across a trading day.
_benchmark_history_cache: dict = {}
_BENCHMARK_HISTORY_CACHE_TTL_SECONDS = 6 * 3600

# yfinance's own sector taxonomy (confirmed via a live fetch across
# Technology/Financial Services/Energy/Healthcare/Consumer Defensive/
# Consumer Cyclical/Utilities/Real Estate/Industrials/Communication
# Services/Basic Materials -- AAPL/JPM/XOM/JNJ/PG/AMZN/NEE/AMT/CAT/
# DIS/LIN) doesn't match GICS's own official sector names exactly (e.g.
# "Financial Services" not "Financials", "Consumer Cyclical" not
# "Consumer Discretionary") -- these keys are the real strings, not
# textbook GICS names. Best-effort SPDR sector-ETF proxy, not a strict
# GICS mapping: a company's true competitive peer set is narrower than
# its whole sector, but there's no peer-multiple data source wired into
# this pipeline (see relative_valuation.py's own docstring) and this is
# the closest available reference without one.
SECTOR_ETF_PROXIES = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Consumer Defensive": "XLP",
    "Consumer Cyclical": "XLY",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Industrials": "XLI",
    "Communication Services": "XLC",
    "Basic Materials": "XLB",
}


def get_benchmark_history(ticker: str, period: str = "5y"):
    """Daily OHLCV history for a benchmark/index/ETF ticker (e.g.
    "^GSPC", "^TNX", a sector ETF) -- returns None on any fetch failure
    rather than raising, since every caller treats a missing benchmark
    series as "that factor degrades to None," not a request-ending
    error (see AlphaFactorsEngine's own degrade-independently
    contract)."""
    cached = _benchmark_history_cache.get(ticker)
    if cached is not None and time.time() - cached[0] < _BENCHMARK_HISTORY_CACHE_TTL_SECONDS:
        return cached[1]

    try:
        df = yf.Ticker(ticker).history(period=period)
    except Exception:
        return None
    if df.empty:
        return None

    _benchmark_history_cache[ticker] = (time.time(), df)
    return df


class MarketDataLoader:

  def __init__(self,ticker:str):
     self.ticker=ticker.upper()
     self.stock=yf.Ticker(self.ticker)
     # Set by get_company_info() below on every call -- True only when
     # its own retry-wrapped .info fetch failed outright (a transient
     # provider issue), never for a fetch that succeeded but simply had
     # nothing useful for this ticker. market_data_tool.py reads this
     # right after calling get_company_info() to tell the two apart.
     self.info_fetch_failed = False

  def get_company_info(self):
     def _do():
        return self.stock.info

     try:
        info = retry_on_transient_error(_do)
     except Exception:
        # Same never-crash-the-caller contract as get_quote() above,
        # just degrading to {} instead of raising -- get_company_info()
        # callers (market_data_tool.py) expect a dict back, not an
        # exception. The failure is still surfaced, just via
        # info_fetch_failed rather than a raised exception, so a
        # throttled/failed fetch isn't silently indistinguishable from
        # yfinance genuinely having nothing for a bad ticker (which
        # returns an empty-ish info dict WITHOUT raising at all).
        self.info_fetch_failed = True
        return {}

     self.info_fetch_failed = False
     return {
        "company_name":info.get("longName"),
        "sector":info.get("sector"),
        "industry":info.get("industry"),
        "market_cap":info.get("marketCap"),
        "currency":info.get("currency"),
        "country":info.get("country"),
        "beta":info.get("beta"),
        "current_price":info.get("currentPrice") or info.get("regularMarketPrice"),
        # Real, already-written business description -- used for the
        # Company Overview report section instead of asking the LLM
        # to invent one from scratch.
        "business_summary":info.get("longBusinessSummary"),
        "website":info.get("website"),
        "employees":info.get("fullTimeEmployees"),
        "next_earnings_date":self.get_next_earnings_date(),
     }

  def get_next_earnings_date(self):
     """
     Nearest upcoming earnings date (datetime.date), or None if
     yfinance doesn't report one or every reported date is already in
     the past. Never raises -- an earnings-calendar miss shouldn't
     break the rest of the report. Used by narrative_builder.py to
     flag an imminent earnings date explicitly rather than letting it
     get lost among the numbered news items (confirmed on a real MSFT
     run, published the day before earnings with the options market
     pricing a 7% move, where the narrative never mentioned it at
     all).
     """
     try:
        dates = (self.stock.calendar or {}).get("Earnings Date") or []
        upcoming = [d for d in dates if d >= date.today()]
        return min(upcoming) if upcoming else None
     except Exception:
        return None

  def get_historical_prices(self,period="5y"):

     df=retry_on_transient_error(lambda: self.stock.history(period=period))
     if df.empty:
        raise ValueError("No price data found for {self.ticker}")
     return df

  def _cached_statement(self, statement_name, fetch_fn, error_message):
     key = make_key("statement", self.ticker, statement_name)
     cached = cache_get(key)
     if cached is not None:
        return cached

     df = retry_on_transient_error(fetch_fn)
     if df.empty:
        raise ValueError(error_message)

     cache_set(key, df, ttl_seconds=STATEMENT_CACHE_TTL_SECONDS)
     return df

  def get_income_statement(self):
     return self._cached_statement(
        "income", lambda: self.stock.financials, "Income Statement unavailable"
     )

  def get_balance_sheet(self):
     return self._cached_statement(
        "balance_sheet", lambda: self.stock.balance_sheet, "Balance Sheet unavailable"
     )

  def get_cash_flow(self):
     return self._cached_statement(
        "cash_flow", lambda: self.stock.cashflow, "Cash flow statement unavailable"
     )

  # Quarterly counterparts -- same caching/error convention as the
  # annual statements above, just pointed at yfinance's quarterly_*
  # accessors (never called anywhere in this app before the stock-detail
  # page's Financial Performance chart/quarterly-yearly toggle needed
  # them). Kept as separate cache keys ("quarterly_income" etc.), not a
  # period arg on the existing methods -- yfinance itself exposes these
  # as distinct properties, and mixing annual/quarterly cache entries
  # under one key would risk serving the wrong shape.
  def get_quarterly_income_statement(self):
     return self._cached_statement(
        "quarterly_income", lambda: self.stock.quarterly_financials, "Quarterly income statement unavailable"
     )

  def get_quarterly_balance_sheet(self):
     return self._cached_statement(
        "quarterly_balance_sheet", lambda: self.stock.quarterly_balance_sheet, "Quarterly balance sheet unavailable"
     )

  def get_quarterly_cash_flow(self):
     return self._cached_statement(
        "quarterly_cash_flow", lambda: self.stock.quarterly_cashflow, "Quarterly cash flow statement unavailable"
     )


  def extract_metric(self,statement_df,metric_name):
      if metric_name in statement_df.index:
          return statement_df.loc[metric_name]
      return None

  def get_all_data(self):
    return{
        "company_info": self.get_company_info(),
        "income_statement": self.get_income_statement(),
        "balance_sheet": self.get_balance_sheet(),
        "cash_flow": self.get_cash_flow(),
        "historical_prices": self.get_historical_prices()
    }
