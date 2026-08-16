"""
options_pricer.py

Black-Scholes-Merton options pricing and Greeks, computed against REAL
live options chain data (yfinance's `.options` expiry list and
`.option_chain(expiry)` -- see MarketDataLoader in app/data/market_data.py)
rather than a synthetic/simulated chain. Powers GET
/v1/stocks/{ticker}/options in app/api/main.py.

No scipy dependency: scipy is NOT in requirements.txt (only present
transitively, e.g. via scikit-learn/chromadb -- fragile to depend on
directly, since a future dependency change could silently drop it).
The standard normal CDF/PDF are instead the closed-form math.erf-based
identity (Phi(x) = 0.5 * (1 + erf(x / sqrt(2))), phi(x) = the usual
Gaussian density) -- exact, not an approximation, just expressed via
stdlib `math` instead of scipy.stats.norm. implied_volatility solves
for sigma with Newton-Raphson (using the option's own vega as the
derivative, the standard approach), falling back to bounded bisection
(the same bracket-and-halve idea as scipy.optimize.brentq, hand-rolled)
if Newton doesn't converge from its starting guess.

Pricing model: standard Black-Scholes-Merton with a continuous
dividend yield `q` (0.0 throughout this module's own orchestrator,
build_options_analysis, below -- MarketDataLoader has no cheap
dividend-yield accessor, and q=0 is a standard, defensible
simplification for a research tool; the `q` parameter is still real
and load-bearing on every pricing/Greeks function for a caller that
does have one).

Every Greek/IV/vol function follows this codebase's existing "degrade
to None on an unrepresentable input, never crash or fabricate a
number" convention (see market_data.py's own docstrings for the same
pattern elsewhere) -- a market price that's below intrinsic value, a
non-positive price, an already-expired contract, or a solver that
simply doesn't converge all degrade to None rather than raising or
returning a nonsense number.

Per-row Greeks in build_options_analysis are derived from an implied
volatility SOLVED against each option's real market price (lastPrice),
not from yfinance's own pre-computed impliedVolatility column -- this
is the honest way to report "here's what the market is actually
pricing in," not a re-statement of a number yfinance already computed
differently (their own IV solver rounds/clips inputs in ways this
module doesn't replicate). yfinance's own IV is still surfaced
alongside as `yfinance_implied_vol_pct` on each row, purely as a
sanity-check field.
"""

import math
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

from app.core.retry import retry_on_transient_error
from app.data.market_data import MarketDataLoader, get_quote
from app.valuation.valuation_pipeline import RISK_FREE_RATE_BY_CURRENCY


class OptionsUnavailableError(Exception):
    """Raised when a ticker has no listed options at all (true for most
    non-US listings, and plenty of small/thinly-traded US names --
    yfinance's own `.options` comes back an empty tuple, it doesn't
    raise), or when a caller-requested `expiry` isn't one yfinance
    actually lists for this ticker. Deliberately kept in this module
    rather than alongside TickerNotFoundError in market_data.py:
    TickerNotFoundError means "this ticker doesn't resolve to real
    market data at all" and is reused across a dozen unrelated tools
    (technicals, financials, insights...); this is a narrower condition
    specific to an otherwise perfectly valid, resolvable ticker simply
    not having a listed options market, so it belongs with the one
    module (this one) that raises and catches it."""


# Skip same-day/next-day expiries when auto-selecting one -- T that
# small makes every Greek (especially theta/gamma, which both blow up
# as T -> 0) numerically unstable and not representative of a normal
# trading day. 5 calendar days is a deliberately generous floor, not a
# tight one: weekly options expire every Friday, so this typically
# still lands on "next week's" expiry, never jumping a full month out.
MIN_DAYS_TO_EXPIRY = 5

DEFAULT_MONEYNESS_BAND_PCT = 20.0

# Implied-vol solver search range -- 1bp to 500% annualized. The upper
# bound is deliberately generous (a genuinely distressed/meme-stock
# name can trade with real implied vols well above 100%); the lower
# bound stops just short of 0 since Black-Scholes divides by
# sigma * sqrt(T) and sigma=0 is a degenerate, zero-vol case a solver
# can't bracket a root against anyway.
_IV_SOLVER_LOW = 1e-4
_IV_SOLVER_HIGH = 5.0
_IV_SOLVER_TOL = 1e-6
_IV_NEWTON_MAX_ITER = 50
_IV_BISECTION_MAX_ITER = 200


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via the exact math.erf identity -- not an
    approximation, just expressed with stdlib `math` instead of
    scipy.stats.norm.cdf (see this module's own docstring for why
    scipy isn't used here)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0) -> float:
    """Standard Black-Scholes-Merton price with continuous dividend
    yield q. `option_type` is "call" or "put". Assumes T > 0 and
    sigma > 0 (callers with an expired contract or degenerate zero-vol
    input should special-case that themselves -- this stays a
    textbook-exact implementation rather than silently substituting
    intrinsic value for an input the formula wasn't asked to handle)."""
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        price = S * math.exp(-q * T) * _normal_cdf(d1) - K * math.exp(-r * T) * _normal_cdf(d2)
    else:
        price = K * math.exp(-r * T) * _normal_cdf(-d2) - S * math.exp(-q * T) * _normal_cdf(-d1)
    return float(price)


def _raw_vega(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    """Annualized, un-rescaled vega (dPrice/dSigma) -- used internally
    as the Newton-Raphson derivative in implied_volatility. The public
    Greek in black_scholes_greeks is this same quantity rescaled to a
    per-1%-vol-move convention (see that function's own docstring)."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return S * math.exp(-q * T) * _normal_pdf(d1) * math.sqrt(T)


def black_scholes_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0) -> dict:
    """Returns {"delta", "gamma", "theta", "vega", "rho"} using the
    standard closed-form Black-Scholes-Merton Greeks. Displayed in the
    conventional trading-desk units, not raw annualized units:
    theta is PER DAY (raw annual theta / 365), vega is per 1% move in
    volatility (raw vega / 100), rho is per 1% move in the risk-free
    rate (raw rho / 100). gamma and delta are already unit-scaled by
    construction (gamma is already "per $1 move in S, per $1 move in
    delta"; delta is already a 0-1 / -1-0 probability-like quantity)."""
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = _normal_pdf(d1)
    disc_q = math.exp(-q * T)
    disc_r = math.exp(-r * T)

    gamma = disc_q * pdf_d1 / (S * sigma * math.sqrt(T))
    vega_annual = S * disc_q * pdf_d1 * math.sqrt(T)

    if option_type == "call":
        delta = disc_q * _normal_cdf(d1)
        theta_annual = (
            -(S * pdf_d1 * sigma * disc_q) / (2 * math.sqrt(T))
            - r * K * disc_r * _normal_cdf(d2)
            + q * S * disc_q * _normal_cdf(d1)
        )
        rho_annual = K * T * disc_r * _normal_cdf(d2)
    else:
        delta = -disc_q * _normal_cdf(-d1)
        theta_annual = (
            -(S * pdf_d1 * sigma * disc_q) / (2 * math.sqrt(T))
            + r * K * disc_r * _normal_cdf(-d2)
            - q * S * disc_q * _normal_cdf(-d1)
        )
        rho_annual = -K * T * disc_r * _normal_cdf(-d2)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta_annual / 365.0),
        "vega": float(vega_annual / 100.0),
        "rho": float(rho_annual / 100.0),
    }


def _bisection(objective, low: float, high: float, tol: float = _IV_SOLVER_TOL, max_iter: int = _IV_BISECTION_MAX_ITER) -> Optional[float]:
    """Bounded bracket-and-halve root find -- the same idea as
    scipy.optimize.brentq, hand-rolled (see this module's own docstring
    for why scipy isn't used here). Callers must have already confirmed
    objective(low) and objective(high) have opposite signs; returns
    None if that's not the case, rather than raising."""
    f_low = objective(low)
    f_high = objective(high)
    if f_low == 0:
        return low
    if f_high == 0:
        return high
    if f_low * f_high > 0:
        return None

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        f_mid = objective(mid)
        if abs(f_mid) < tol or (high - low) / 2.0 < tol:
            return mid
        if f_low * f_mid < 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2.0


def implied_volatility(
    market_price: Optional[float], S: float, K: float, T: float, r: float, option_type: str, q: float = 0.0
) -> Optional[float]:
    """Solves for the sigma that makes black_scholes_price(...) equal
    market_price: Newton-Raphson first (using the option's own vega as
    the derivative, starting from an at-the-money-ish sigma=0.3 guess),
    falling back to bounded bisection over (_IV_SOLVER_LOW,
    _IV_SOLVER_HIGH) if Newton fails to converge or steps outside that
    range. Returns None -- never raises -- for any input the model
    can't represent: an expired contract (T <= 0), a non-positive
    market price, a market price below the option's own intrinsic
    value (no finite sigma satisfies Black-Scholes there, since the
    model's price is always >= intrinsic value for sigma > 0), or a
    solver that simply fails to converge."""
    if option_type not in ("call", "put"):
        return None
    if market_price is None or market_price <= 0:
        return None
    if T <= 0:
        return None
    if S <= 0 or K <= 0:
        return None

    if option_type == "call":
        intrinsic = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    else:
        intrinsic = max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
    if market_price < intrinsic:
        return None

    def _objective(sigma: float) -> float:
        return black_scholes_price(S, K, T, r, sigma, option_type, q) - market_price

    sigma = 0.3
    converged = False
    try:
        for _ in range(_IV_NEWTON_MAX_ITER):
            f = _objective(sigma)
            if abs(f) < _IV_SOLVER_TOL:
                converged = True
                break
            vega = _raw_vega(S, K, T, r, sigma, q)
            if vega <= 1e-12:
                break
            sigma_next = sigma - f / vega
            if not math.isfinite(sigma_next) or sigma_next <= _IV_SOLVER_LOW or sigma_next >= _IV_SOLVER_HIGH:
                break
            sigma = sigma_next
        else:
            converged = abs(_objective(sigma)) < _IV_SOLVER_TOL
    except (ValueError, OverflowError, ZeroDivisionError):
        converged = False

    if converged:
        return float(sigma)

    # Newton didn't land -- fall back to bounded bisection, which only
    # needs the two endpoints to bracket a sign change (no derivative,
    # so it can't diverge the way Newton can from a bad starting point).
    try:
        sigma = _bisection(_objective, _IV_SOLVER_LOW, _IV_SOLVER_HIGH)
    except (ValueError, OverflowError, ZeroDivisionError):
        return None

    return float(sigma) if sigma is not None else None


def realized_volatility(close_prices: pd.Series, trading_days_per_year: int = 252) -> Optional[float]:
    """Annualized realized volatility from daily log returns, as a
    PERCENTAGE (e.g. 32.4, not 0.324): std(log_returns) *
    sqrt(trading_days_per_year) * 100. Returns None (never raises) for
    fewer than 2 usable price points, or if the standard deviation
    itself comes back undefined (e.g. a single log return, which has
    no sample variance)."""
    if close_prices is None or len(close_prices) < 2:
        return None

    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    if log_returns.empty:
        return None

    std = log_returns.std()
    if pd.isna(std):
        return None

    return float(std * math.sqrt(trading_days_per_year) * 100)


def _detect_currency(ticker: str) -> str:
    """Same ticker-suffix convention used throughout this codebase
    (see valuation_tool.py/report_data_builder.py/rag_pipeline.py's own
    `.endswith(".NS")` checks) -- a ".NS" (NSE) ticker is INR, every
    other ticker defaults to USD. Not derived from yfinance's own
    fast_info currency field here, to stay consistent with how every
    other ticker-scoped tool in this app already decides USD vs. INR."""
    return "INR" if ticker.upper().endswith(".NS") else "USD"


def _select_expiry(expiries: list, requested: Optional[str]) -> str:
    if requested is not None:
        if requested not in expiries:
            raise OptionsUnavailableError(
                f"'{requested}' is not a listed expiry for this ticker. "
                f"Available expiries: {', '.join(expiries)}"
            )
        return requested

    today = date.today()
    for expiry_str in expiries:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        if (expiry_date - today).days >= MIN_DAYS_TO_EXPIRY:
            return expiry_str
    # Nothing cleared the 5-day floor (every listed expiry is imminent)
    # -- fall back to the single nearest one rather than refusing to
    # return anything at all.
    return expiries[0]


def _filter_by_moneyness(chain_df: pd.DataFrame, spot: float, band_pct: float) -> pd.DataFrame:
    low = spot * (1 - band_pct / 100.0)
    high = spot * (1 + band_pct / 100.0)
    return chain_df[(chain_df["strike"] >= low) & (chain_df["strike"] <= high)]


def _safe_float(value) -> Optional[float]:
    return float(value) if pd.notna(value) else None


def _safe_int(value) -> Optional[int]:
    return int(value) if pd.notna(value) else None


def _row_to_option_dict(row, S: float, T: float, r: float, q: float, option_type: str) -> dict:
    strike = float(row["strike"])
    market_price = float(row["lastPrice"])

    yfinance_iv = row.get("impliedVolatility")
    yfinance_implied_vol_pct = float(yfinance_iv) * 100 if pd.notna(yfinance_iv) else None

    iv = implied_volatility(market_price, S, strike, T, r, option_type, q)

    implied_vol_pct = None
    theoretical_price = None
    greeks = {"delta": None, "gamma": None, "theta": None, "vega": None, "rho": None}
    if iv is not None:
        implied_vol_pct = float(iv * 100)
        theoretical_price = black_scholes_price(S, strike, T, r, iv, option_type, q)
        greeks = black_scholes_greeks(S, strike, T, r, iv, option_type, q)

    in_the_money = bool(row["inTheMoney"]) if "inTheMoney" in row and pd.notna(row.get("inTheMoney")) else (
        S > strike if option_type == "call" else S < strike
    )

    return {
        "strike": strike,
        "market_price": market_price,
        "bid": _safe_float(row.get("bid")),
        "ask": _safe_float(row.get("ask")),
        "implied_vol_pct": implied_vol_pct,
        "yfinance_implied_vol_pct": yfinance_implied_vol_pct,
        "theoretical_price": theoretical_price,
        "delta": greeks["delta"],
        "gamma": greeks["gamma"],
        "theta": greeks["theta"],
        "vega": greeks["vega"],
        "rho": greeks["rho"],
        "in_the_money": in_the_money,
        "open_interest": _safe_int(row.get("openInterest")),
    }


def build_options_analysis(ticker: str, expiry: Optional[str] = None, moneyness_band_pct: float = DEFAULT_MONEYNESS_BAND_PCT) -> dict:
    """Orchestrator behind GET /v1/stocks/{ticker}/options. Raises
    TickerNotFoundError (via get_quote) for a bad/delisted ticker, and
    OptionsUnavailableError for a valid ticker with no listed options
    market or an `expiry` that isn't one yfinance actually lists."""
    ticker = ticker.upper()
    loader = MarketDataLoader(ticker)

    # get_quote() already raises TickerNotFoundError for a bad/delisted
    # ticker -- reused as-is rather than re-deriving spot price from
    # MarketDataLoader.get_company_info() (two extra round trips for
    # dozens of unused fields, the same reasoning get_quote's own
    # docstring gives for existing elsewhere).
    quote = get_quote(ticker)
    S = float(quote["price"])

    try:
        expiries = list(retry_on_transient_error(lambda: loader.stock.options))
    except Exception as exc:
        raise OptionsUnavailableError(f"No listed options found for {ticker}.") from exc
    if not expiries:
        raise OptionsUnavailableError(f"No listed options found for {ticker}.")

    selected_expiry = _select_expiry(expiries, expiry)

    chain = retry_on_transient_error(lambda: loader.stock.option_chain(selected_expiry))

    expiry_date = datetime.strptime(selected_expiry, "%Y-%m-%d").date()
    days_to_expiry = (expiry_date - date.today()).days
    T = days_to_expiry / 365.0

    currency = _detect_currency(ticker)
    r = RISK_FREE_RATE_BY_CURRENCY.get(currency, RISK_FREE_RATE_BY_CURRENCY["USD"])

    # A realized-vol fetch failure (e.g. a brand-new listing with under
    # a year of history) degrades to None -- it's a useful sanity-check
    # field alongside the market-implied vols below, not a hard
    # dependency of the options chain itself.
    realized_volatility_pct = None
    try:
        history = loader.get_historical_prices(period="1y")
        realized_volatility_pct = realized_volatility(history["Close"])
    except Exception:
        realized_volatility_pct = None

    # No cheap continuous-dividend-yield accessor on MarketDataLoader --
    # see this module's own docstring for why q=0.0 is used here.
    q = 0.0

    calls_df = _filter_by_moneyness(chain.calls, S, moneyness_band_pct)
    puts_df = _filter_by_moneyness(chain.puts, S, moneyness_band_pct)

    calls = sorted(
        (_row_to_option_dict(row, S, T, r, q, "call") for _, row in calls_df.iterrows()),
        key=lambda o: o["strike"],
    )
    puts = sorted(
        (_row_to_option_dict(row, S, T, r, q, "put") for _, row in puts_df.iterrows()),
        key=lambda o: o["strike"],
    )

    return {
        "ticker": ticker,
        "spot_price": S,
        "currency": currency,
        "risk_free_rate": r,
        "realized_volatility_pct": realized_volatility_pct,
        "expiries": expiries,
        "selected_expiry": selected_expiry,
        "days_to_expiry": days_to_expiry,
        "calls": calls,
        "puts": puts,
    }
