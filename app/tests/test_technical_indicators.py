"""
Unit tests for app/analysis/technical_indicators.py -- synthetic OHLCV
DataFrames (yfinance's own `.history()` shape: Open/High/Low/Close/
Volume columns, DatetimeIndex), no network. Indicator-level tests check
documented mathematical properties (bounds, warm-up NaN behavior, sign
under a known trend) rather than hand-verified magic numbers, matching
this app's existing "test the contract" convention for formula-heavy
code (see test_dcf_engine.py's own style).
"""

import json

import pandas as pd

from app.analysis import technical_indicators as ti
from app.data.market_data import MarketDataLoader


def _ohlcv(closes, start="2024-01-01"):
    """Builds a plausible OHLCV frame from a list of closing prices --
    High/Low bracket the close by a fixed small margin, Open equals the
    prior close (both harmless simplifications for testing formulas
    that only really care about Close, plus High/Low for ADX/Stochastic/
    Williams %R)."""
    index = pd.date_range(start, periods=len(closes), freq="D")
    closes = pd.Series(closes, index=index, dtype=float)
    return pd.DataFrame({
        "Open": closes.shift(1).fillna(closes.iloc[0]),
        "High": closes + 1.0,
        "Low": closes - 1.0,
        "Close": closes,
        "Volume": pd.Series([1_000_000] * len(closes), index=index),
    })


def _uptrend(n=250, start=100.0, step=0.5):
    return _ohlcv([start + i * step for i in range(n)])


def _downtrend(n=250, start=300.0, step=0.5):
    return _ohlcv([start - i * step for i in range(n)])


# ---------------------------------------------------------------- RSI

def test_rsi_is_bounded_0_to_100():
    df = _uptrend(n=60)
    rsi = ti._rsi(df["Close"])
    valid = rsi.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_approaches_100_on_a_pure_uptrend():
    # Every day is a gain, no losses -- RSI should sit near the top of
    # its range once warmed up.
    df = _uptrend(n=60, step=1.0)
    rsi = ti._rsi(df["Close"])
    assert rsi.iloc[-1] > 95


def test_rsi_is_nan_during_warmup():
    df = _uptrend(n=10)
    rsi = ti._rsi(df["Close"])
    assert pd.isna(rsi.iloc[0])


# ---------------------------------------------------------------- MACD

def test_macd_histogram_is_positive_on_a_sustained_uptrend():
    df = _uptrend(n=100)
    _, _, histogram = ti._macd(df["Close"])
    assert histogram.iloc[-1] > 0


def test_macd_histogram_is_negative_on_a_sustained_downtrend():
    df = _downtrend(n=100)
    _, _, histogram = ti._macd(df["Close"])
    assert histogram.iloc[-1] < 0


# ---------------------------------------------------------------- ADX

def test_adx_is_non_negative():
    df = _uptrend(n=60)
    adx = ti._adx(df["High"], df["Low"], df["Close"])
    valid = adx.dropna()
    assert (valid >= 0).all()


# ---------------------------------------------------------------- VWAP

def test_vwap_falls_between_low_and_high_of_the_full_window():
    df = _uptrend(n=30)
    vwap = ti._vwap(df["High"], df["Low"], df["Close"], df["Volume"])
    assert df["Low"].min() <= vwap.iloc[-1] <= df["High"].max()


# ---------------------------------------------------------------- Bollinger

def test_bollinger_upper_is_always_above_lower():
    df = _uptrend(n=60)
    upper, lower = ti._bollinger(df["Close"])
    valid = pd.concat([upper, lower], axis=1).dropna()
    assert (valid.iloc[:, 0] >= valid.iloc[:, 1]).all()


# ---------------------------------------------------------------- Stochastic / Williams %R

def test_stochastic_k_is_bounded_0_to_100():
    df = _uptrend(n=60)
    k, _ = ti._stochastic(df["High"], df["Low"], df["Close"])
    valid = k.dropna()
    assert (valid >= -0.01).all() and (valid <= 100.01).all()


def test_williams_r_is_bounded_negative_100_to_0():
    df = _uptrend(n=60)
    wr = ti._williams_r(df["High"], df["Low"], df["Close"])
    valid = wr.dropna()
    assert (valid >= -100.01).all() and (valid <= 0.01).all()


# ---------------------------------------------------------------- build_technicals (orchestration)

def test_build_technicals_reports_bullish_trend_on_a_sustained_uptrend(monkeypatch):
    df = _uptrend(n=250)
    monkeypatch.setattr(MarketDataLoader, "__init__", lambda self, t: setattr(self, "ticker", t.upper()))
    monkeypatch.setattr(MarketDataLoader, "get_historical_prices", lambda self, period="5y": df)

    result = ti.build_technicals("TEST", range_period="1y")

    assert result["trend"] == "Bullish"
    assert result["moving_average_signal"]["verdict"] == "Buy"


def test_build_technicals_reports_bearish_trend_on_a_sustained_downtrend(monkeypatch):
    df = _downtrend(n=250)
    monkeypatch.setattr(MarketDataLoader, "__init__", lambda self, t: setattr(self, "ticker", t.upper()))
    monkeypatch.setattr(MarketDataLoader, "get_historical_prices", lambda self, period="5y": df)

    result = ti.build_technicals("TEST", range_period="1y")

    assert result["trend"] == "Bearish"
    assert result["moving_average_signal"]["verdict"] == "Sell"


def test_build_technicals_returns_price_history_matching_input_length(monkeypatch):
    df = _uptrend(n=40)
    monkeypatch.setattr(MarketDataLoader, "__init__", lambda self, t: setattr(self, "ticker", t.upper()))
    monkeypatch.setattr(MarketDataLoader, "get_historical_prices", lambda self, period="5y": df)

    result = ti.build_technicals("TEST", range_period="1y")

    assert len(result["price_history"]) == 40
    assert result["price_history"][0]["date"] == "2024-01-01"
    assert set(result["price_history"][0].keys()) == {"date", "open", "high", "low", "close", "volume"}


def test_build_technicals_overlay_series_are_date_aligned_with_price_history(monkeypatch):
    df = _uptrend(n=40)
    monkeypatch.setattr(MarketDataLoader, "__init__", lambda self, t: setattr(self, "ticker", t.upper()))
    monkeypatch.setattr(MarketDataLoader, "get_historical_prices", lambda self, period="5y": df)

    result = ti.build_technicals("TEST", range_period="1y")

    price_dates = [row["date"] for row in result["price_history"]]
    ema_dates = [point[0] for point in result["overlays"]["ema_20"]]
    assert price_dates == ema_dates


def test_build_technicals_drops_rows_with_missing_ohlc_and_stays_json_safe(monkeypatch):
    # Regression: real yfinance history can include a row with a NaN
    # Open/High/Low/Close (e.g. an in-progress current trading day) --
    # a bare NaN float reaching FastAPI's JSONResponse crashes the
    # whole request (Starlette's json.dumps uses allow_nan=False), so
    # that row must be dropped, not passed through as NaN.
    df = _uptrend(n=10)
    df.loc[df.index[5], "Open"] = float("nan")

    monkeypatch.setattr(MarketDataLoader, "__init__", lambda self, t: setattr(self, "ticker", t.upper()))
    monkeypatch.setattr(MarketDataLoader, "get_historical_prices", lambda self, period="5y": df)

    result = ti.build_technicals("TEST", range_period="1y")

    assert len(result["price_history"]) == 9
    dropped_date = df.index[5].date().isoformat()
    assert dropped_date not in [row["date"] for row in result["price_history"]]
    assert dropped_date not in [point[0] for point in result["overlays"]["ema_20"]]
    for row in result["price_history"]:
        assert all(v == v for v in (row["open"], row["high"], row["low"], row["close"]))  # v == v is False only for NaN
    json.dumps(result, allow_nan=False)  # must not raise


def test_build_technicals_support_resistance_uses_swing_windows(monkeypatch):
    df = _uptrend(n=100)
    monkeypatch.setattr(MarketDataLoader, "__init__", lambda self, t: setattr(self, "ticker", t.upper()))
    monkeypatch.setattr(MarketDataLoader, "get_historical_prices", lambda self, period="5y": df)

    result = ti.build_technicals("TEST", range_period="1y")

    assert result["support_resistance"]["support_20d"] < result["support_resistance"]["resistance_20d"]
    assert result["support_resistance"]["support_60d"] < result["support_resistance"]["resistance_60d"]


def test_build_technicals_raises_on_empty_price_history(monkeypatch):
    def _raise(self, period="5y"):
        raise ValueError("No price data found")

    monkeypatch.setattr(MarketDataLoader, "__init__", lambda self, t: setattr(self, "ticker", t.upper()))
    monkeypatch.setattr(MarketDataLoader, "get_historical_prices", _raise)

    try:
        ti.build_technicals("TEST", range_period="1y")
        assert False, "expected ValueError"
    except ValueError:
        pass
