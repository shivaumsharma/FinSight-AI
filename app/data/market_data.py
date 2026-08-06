import time
from datetime import date

import yfinance as yf
import pandas as pd

from app.core.cache import cache_get, cache_set, make_key

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
    "change_pct"}. Raises TickerNotFoundError if fast_info has no
    usable price (bad/delisted symbol), same exception the rest of this
    module already uses for that condition."""
    ticker = ticker.upper()

    cached = _quote_cache.get(ticker)
    if cached is not None and time.time() - cached[0] < _QUOTE_CACHE_TTL_SECONDS:
        return cached[1]

    try:
        fast_info = yf.Ticker(ticker).fast_info
        price = fast_info.last_price
        previous_close = fast_info.previous_close
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
    quote = {"price": price, "change_pct": change_pct}
    _quote_cache[ticker] = (time.time(), quote)
    return quote


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
     
     df=self.stock.history(period=period)
     if df.empty:
        raise ValueError("No price data found for {self.ticker}")
     return df
  
  def _cached_statement(self, statement_name, fetch_fn, error_message):
     key = make_key("statement", self.ticker, statement_name)
     cached = cache_get(key)
     if cached is not None:
        return cached

     df = fetch_fn()
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

   