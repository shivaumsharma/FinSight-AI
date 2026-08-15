"""
Unit tests for app/derivatives/options_pricer.py -- Black-Scholes-Merton
pricing/Greeks are checked against known textbook reference values and
documented mathematical properties (put-call parity, delta bounds,
call/put gamma equality), matching this app's existing "test the
contract" convention for formula-heavy code (see test_dcf_engine.py's
own style). build_options_analysis is tested with MarketDataLoader/
yfinance mocked, following the same "monkeypatch MarketDataLoader.__init__
plus the method(s) actually used" convention as
test_technical_indicators.py's build_technicals tests, and the same
"main.py/other tools patch their own separately-bound get_quote name"
convention documented in test_watchlist.py (options_pricer.py does
`from app.data.market_data import ... get_quote`, so tests here patch
options_pricer.get_quote directly, not market_data.get_quote).
"""

import math
from datetime import date, timedelta

import pandas as pd
import pytest

import app.derivatives.options_pricer as op
from app.data.market_data import MarketDataLoader

# ---------------------------------------------------------------- black_scholes_price

def test_black_scholes_price_matches_textbook_reference_values():
    # Standard reference case (S=100, K=100, T=1, r=0.05, sigma=0.2,
    # q=0): call ~= 10.4506, put ~= 5.5735.
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    call = op.black_scholes_price(S, K, T, r, sigma, "call")
    put = op.black_scholes_price(S, K, T, r, sigma, "put")
    assert call == pytest.approx(10.4506, abs=0.01)
    assert put == pytest.approx(5.5735, abs=0.01)


def test_put_call_parity_holds():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    call = op.black_scholes_price(S, K, T, r, sigma, "call")
    put = op.black_scholes_price(S, K, T, r, sigma, "put")
    assert call - put == pytest.approx(S - K * math.exp(-r * T), abs=0.01)


def test_black_scholes_price_rejects_unknown_option_type():
    with pytest.raises(ValueError):
        op.black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, "straddle")


# ---------------------------------------------------------------- black_scholes_greeks

def test_delta_is_near_one_for_a_deep_in_the_money_call():
    greeks = op.black_scholes_greeks(200.0, 100.0, 0.5, 0.05, 0.2, "call")
    assert greeks["delta"] > 0.95


def test_delta_is_near_zero_for_a_deep_out_of_the_money_call():
    greeks = op.black_scholes_greeks(50.0, 100.0, 0.5, 0.05, 0.2, "call")
    assert greeks["delta"] < 0.05


def test_gamma_is_identical_for_a_call_and_put_at_the_same_strike_and_expiry():
    # A real, checkable Black-Scholes property: gamma doesn't depend on
    # option_type, only on S/K/T/r/sigma/q.
    call_greeks = op.black_scholes_greeks(100.0, 100.0, 1.0, 0.05, 0.2, "call")
    put_greeks = op.black_scholes_greeks(100.0, 100.0, 1.0, 0.05, 0.2, "put")
    assert call_greeks["gamma"] == pytest.approx(put_greeks["gamma"], abs=1e-9)


def test_greeks_returns_all_five_expected_keys():
    greeks = op.black_scholes_greeks(100.0, 100.0, 1.0, 0.05, 0.2, "call")
    assert set(greeks.keys()) == {"delta", "gamma", "theta", "vega", "rho"}
    assert all(isinstance(v, float) for v in greeks.values())


# ---------------------------------------------------------------- implied_volatility

def test_implied_volatility_round_trip_recovers_the_known_sigma():
    S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.03, 0.30
    price = op.black_scholes_price(S, K, T, r, sigma, "call")
    recovered = op.implied_volatility(price, S, K, T, r, "call")
    assert recovered == pytest.approx(0.30, abs=0.005)


def test_implied_volatility_round_trip_works_for_puts_too():
    S, K, T, r, sigma = 100.0, 110.0, 0.75, 0.04, 0.45
    price = op.black_scholes_price(S, K, T, r, sigma, "put")
    recovered = op.implied_volatility(price, S, K, T, r, "put")
    assert recovered == pytest.approx(0.45, abs=0.005)


def test_implied_volatility_returns_none_for_a_zero_market_price():
    assert op.implied_volatility(0.0, 100.0, 100.0, 1.0, 0.05, "call") is None


def test_implied_volatility_returns_none_for_an_expired_contract():
    assert op.implied_volatility(5.0, 100.0, 100.0, 0.0, 0.05, "call") is None


def test_implied_volatility_returns_none_for_a_price_below_intrinsic_value():
    # A deep ITM call (S=200, K=100) has intrinsic value near 100 --
    # quoting it at 1.0 is unrepresentable by any positive sigma.
    assert op.implied_volatility(1.0, 200.0, 100.0, 1.0, 0.05, "call") is None


def test_implied_volatility_never_raises_on_a_negative_price():
    assert op.implied_volatility(-5.0, 100.0, 100.0, 1.0, 0.05, "call") is None


# ---------------------------------------------------------------- realized_volatility

def test_realized_volatility_matches_a_hand_computed_value():
    # Prices alternate 100 <-> 110, giving a constant-magnitude daily
    # log return of +/- ln(1.1) -- a simple, independently verifiable
    # case: mean return is exactly 0, so sample variance reduces to
    # sum(r^2) / (n - 1).
    closes = pd.Series([100.0, 110.0, 100.0, 110.0, 100.0])
    r = math.log(1.1)
    returns = [r, -r, r, -r]
    variance = sum(x ** 2 for x in returns) / (len(returns) - 1)  # mean is 0
    expected = math.sqrt(variance) * math.sqrt(252) * 100

    result = op.realized_volatility(closes)
    assert result == pytest.approx(expected, abs=1e-9)


def test_realized_volatility_returns_none_for_a_single_price():
    assert op.realized_volatility(pd.Series([100.0])) is None


def test_realized_volatility_returns_none_for_an_empty_series():
    assert op.realized_volatility(pd.Series([], dtype=float)) is None


# ---------------------------------------------------------------- build_options_analysis

class _FakeChain:
    def __init__(self, calls, puts):
        self.calls = calls
        self.puts = puts


class _FakeStock:
    def __init__(self, options, chain=None):
        self.options = options
        self._chain = chain

    def option_chain(self, expiry):
        return self._chain


_CHAIN_COLUMNS = ["strike", "lastPrice", "bid", "ask", "impliedVolatility", "openInterest", "inTheMoney"]


def _empty_chain_df():
    return pd.DataFrame(columns=_CHAIN_COLUMNS)


def _patch_loader(monkeypatch, fake_stock, history_df=None):
    def _init(self, t):
        self.ticker = t.upper()
        self.stock = fake_stock

    monkeypatch.setattr(MarketDataLoader, "__init__", _init)

    if history_df is not None:
        monkeypatch.setattr(MarketDataLoader, "get_historical_prices", lambda self, period="5y": history_df)
    else:
        def _raise(self, period="5y"):
            raise ValueError("no price history")

        monkeypatch.setattr(MarketDataLoader, "get_historical_prices", _raise)


def _history_df(n=260, start=100.0):
    index = pd.date_range("2024-01-01", periods=n, freq="D")
    closes = pd.Series([start + i * 0.01 for i in range(n)], index=index)
    return pd.DataFrame({"Close": closes})


def _fake_quote(price=100.0, currency="USD"):
    return {"price": price, "change_pct": 0.1, "previous_close": price * 0.99, "currency": currency}


def test_build_options_analysis_returns_the_exact_top_level_shape(monkeypatch):
    S = 100.0
    r = 0.04
    sigma = 0.25
    expiry_str = (date.today() + timedelta(days=30)).isoformat()
    T = 30 / 365.0

    call_100 = op.black_scholes_price(S, 100.0, T, r, sigma, "call")
    call_110 = op.black_scholes_price(S, 110.0, T, r, sigma, "call")
    put_100 = op.black_scholes_price(S, 100.0, T, r, sigma, "put")
    put_90 = op.black_scholes_price(S, 90.0, T, r, sigma, "put")

    # Deliberately NOT strike-sorted going in, so the sort-by-strike
    # assertion below is actually exercising something.
    calls_df = pd.DataFrame([
        {"strike": 110.0, "lastPrice": call_110, "bid": call_110 - 0.05, "ask": call_110 + 0.05,
         "impliedVolatility": sigma, "openInterest": 500, "inTheMoney": False},
        {"strike": 100.0, "lastPrice": call_100, "bid": call_100 - 0.05, "ask": call_100 + 0.05,
         "impliedVolatility": sigma, "openInterest": 1000, "inTheMoney": False},
    ])
    puts_df = pd.DataFrame([
        {"strike": 100.0, "lastPrice": put_100, "bid": put_100 - 0.05, "ask": put_100 + 0.05,
         "impliedVolatility": sigma, "openInterest": 800, "inTheMoney": False},
        {"strike": 90.0, "lastPrice": put_90, "bid": put_90 - 0.05, "ask": put_90 + 0.05,
         "impliedVolatility": sigma, "openInterest": 300, "inTheMoney": True},
    ])

    fake_stock = _FakeStock(options=[expiry_str], chain=_FakeChain(calls_df, puts_df))
    _patch_loader(monkeypatch, fake_stock, history_df=_history_df())
    monkeypatch.setattr(op, "get_quote", lambda ticker: _fake_quote(price=S))

    result = op.build_options_analysis("TEST")

    assert set(result.keys()) == {
        "ticker", "spot_price", "currency", "risk_free_rate", "realized_volatility_pct",
        "expiries", "selected_expiry", "days_to_expiry", "calls", "puts",
    }
    assert result["ticker"] == "TEST"
    assert result["spot_price"] == pytest.approx(S)
    assert result["currency"] == "USD"
    assert result["risk_free_rate"] == pytest.approx(0.04)
    assert result["expiries"] == [expiry_str]
    assert result["selected_expiry"] == expiry_str
    assert result["days_to_expiry"] == 30

    call_strikes = [c["strike"] for c in result["calls"]]
    assert call_strikes == sorted(call_strikes) == [100.0, 110.0]
    put_strikes = [p["strike"] for p in result["puts"]]
    assert put_strikes == sorted(put_strikes) == [90.0, 100.0]

    # Priced against real market prices that were themselves generated
    # with sigma=0.25 baked in (via black_scholes_price above, not
    # circular -- the row's lastPrice is the market data, the solver
    # has to recover sigma from that price independently) -- the
    # recovered implied_vol_pct should land close to 25%.
    for row in result["calls"] + result["puts"]:
        assert row["implied_vol_pct"] is not None
        assert row["implied_vol_pct"] == pytest.approx(sigma * 100, abs=1.0)
        assert row["theoretical_price"] is not None
        assert row["delta"] is not None


def test_build_options_analysis_filters_out_of_band_strikes(monkeypatch):
    S = 100.0
    expiry_str = (date.today() + timedelta(days=30)).isoformat()
    # 150 is 50% above spot -- outside the default 20% moneyness band.
    calls_df = pd.DataFrame([
        {"strike": 100.0, "lastPrice": 5.0, "bid": 4.9, "ask": 5.1, "impliedVolatility": 0.2, "openInterest": 100, "inTheMoney": False},
        {"strike": 150.0, "lastPrice": 0.1, "bid": 0.05, "ask": 0.15, "impliedVolatility": 0.2, "openInterest": 10, "inTheMoney": False},
    ])
    puts_df = _empty_chain_df()

    fake_stock = _FakeStock(options=[expiry_str], chain=_FakeChain(calls_df, puts_df))
    _patch_loader(monkeypatch, fake_stock, history_df=_history_df())
    monkeypatch.setattr(op, "get_quote", lambda ticker: _fake_quote(price=S))

    result = op.build_options_analysis("TEST")
    assert [c["strike"] for c in result["calls"]] == [100.0]


def test_build_options_analysis_keeps_a_row_with_none_iv_when_price_is_unusable(monkeypatch):
    expiry_str = (date.today() + timedelta(days=30)).isoformat()
    calls_df = pd.DataFrame([
        {"strike": 100.0, "lastPrice": 0.0, "bid": 0.0, "ask": 0.0, "impliedVolatility": 0.0, "openInterest": 0, "inTheMoney": False},
    ])
    puts_df = _empty_chain_df()

    fake_stock = _FakeStock(options=[expiry_str], chain=_FakeChain(calls_df, puts_df))
    _patch_loader(monkeypatch, fake_stock, history_df=_history_df())
    monkeypatch.setattr(op, "get_quote", lambda ticker: _fake_quote(price=100.0))

    result = op.build_options_analysis("TEST")

    assert len(result["calls"]) == 1
    row = result["calls"][0]
    assert row["implied_vol_pct"] is None
    assert row["theoretical_price"] is None
    assert row["delta"] is None
    assert row["gamma"] is None
    assert row["theta"] is None
    assert row["vega"] is None
    assert row["rho"] is None
    # market_price/strike themselves are never dropped, only the
    # derived fields degrade to None.
    assert row["strike"] == 100.0
    assert row["market_price"] == 0.0


def test_build_options_analysis_uses_inr_risk_free_rate_for_ns_ticker(monkeypatch):
    expiry_str = (date.today() + timedelta(days=30)).isoformat()
    fake_stock = _FakeStock(options=[expiry_str], chain=_FakeChain(_empty_chain_df(), _empty_chain_df()))
    _patch_loader(monkeypatch, fake_stock, history_df=_history_df())
    monkeypatch.setattr(op, "get_quote", lambda ticker: _fake_quote(price=1000.0, currency="INR"))

    result = op.build_options_analysis("RELIANCE.NS")
    assert result["currency"] == "INR"
    assert result["risk_free_rate"] == pytest.approx(0.07)


def test_build_options_analysis_defaults_to_the_nearest_expiry_at_least_5_days_out(monkeypatch):
    too_soon = (date.today() + timedelta(days=2)).isoformat()
    just_right = (date.today() + timedelta(days=10)).isoformat()
    later = (date.today() + timedelta(days=45)).isoformat()

    fake_stock = _FakeStock(
        options=[too_soon, just_right, later],
        chain=_FakeChain(_empty_chain_df(), _empty_chain_df()),
    )
    _patch_loader(monkeypatch, fake_stock, history_df=_history_df())
    monkeypatch.setattr(op, "get_quote", lambda ticker: _fake_quote())

    result = op.build_options_analysis("TEST")
    assert result["selected_expiry"] == just_right


def test_build_options_analysis_falls_back_to_nearest_expiry_when_none_clear_the_floor(monkeypatch):
    only_imminent = (date.today() + timedelta(days=1)).isoformat()
    fake_stock = _FakeStock(options=[only_imminent], chain=_FakeChain(_empty_chain_df(), _empty_chain_df()))
    _patch_loader(monkeypatch, fake_stock, history_df=_history_df())
    monkeypatch.setattr(op, "get_quote", lambda ticker: _fake_quote())

    result = op.build_options_analysis("TEST")
    assert result["selected_expiry"] == only_imminent


def test_build_options_analysis_raises_options_unavailable_for_an_unlisted_expiry(monkeypatch):
    expiry_str = (date.today() + timedelta(days=30)).isoformat()
    fake_stock = _FakeStock(options=[expiry_str], chain=_FakeChain(_empty_chain_df(), _empty_chain_df()))
    _patch_loader(monkeypatch, fake_stock, history_df=_history_df())
    monkeypatch.setattr(op, "get_quote", lambda ticker: _fake_quote())

    with pytest.raises(op.OptionsUnavailableError):
        op.build_options_analysis("TEST", expiry="2099-01-01")


def test_build_options_analysis_raises_options_unavailable_when_no_expiries_listed(monkeypatch):
    # The exact "no listed options market" case this whole feature has
    # to handle gracefully -- many tickers, especially non-US ones,
    # simply have no options chain at all.
    fake_stock = _FakeStock(options=[])
    _patch_loader(monkeypatch, fake_stock, history_df=_history_df())
    monkeypatch.setattr(op, "get_quote", lambda ticker: _fake_quote())

    with pytest.raises(op.OptionsUnavailableError):
        op.build_options_analysis("NOOPTIONS")


def test_build_options_analysis_degrades_realized_volatility_to_none_on_history_failure(monkeypatch):
    expiry_str = (date.today() + timedelta(days=30)).isoformat()
    fake_stock = _FakeStock(options=[expiry_str], chain=_FakeChain(_empty_chain_df(), _empty_chain_df()))
    _patch_loader(monkeypatch, fake_stock, history_df=None)  # get_historical_prices raises
    monkeypatch.setattr(op, "get_quote", lambda ticker: _fake_quote())

    result = op.build_options_analysis("TEST")
    assert result["realized_volatility_pct"] is None
