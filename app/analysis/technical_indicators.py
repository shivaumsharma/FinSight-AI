"""
technical_indicators.py

Powers the stock-detail-page's Technicals tab and price chart -- see
GET /v1/stocks/{ticker}/technicals in app/api/main.py.

Every indicator here is computed from scratch against pandas (no
pandas-ta/ta-lib dependency added -- these are standard, well-documented
formulas and the app already depends on pandas for everything else) over
`MarketDataLoader.get_historical_prices()`'s daily OHLCV DataFrame, which
was already being fetched elsewhere in this app but never used to
compute indicators.

Scope note: only DAILY-bar ranges are supported (1mo/3mo/6mo/1y/2y/5y/
max, matching yfinance's own `history(period=...)` values) -- true
intraday 1D/1W views would need minute/hourly interval data, a genuinely
different fetch shape (and a much larger payload for what this research
tool needs). Not built this pass; see the approved stock-detail-page
plan.

VWAP here is a cumulative volume-weighted average price over the whole
fetched window (typical_price * volume, cumulative-summed) -- the
standard simplification used on daily-bar charts, not a true intraday
VWAP (which needs tick/minute data this app doesn't fetch).

"Support/resistance" is explicitly labeled "recent swing levels" (the
rolling N-day low/high), not "pivot points" -- there's no vendor pivot
methodology behind these numbers, and claiming "pivot points" would
imply a specific (unimplemented) calculation this isn't.
"""

from typing import Optional

import numpy as np
import pandas as pd

from app.data.market_data import MarketDataLoader

RSI_PERIOD = 14
ADX_PERIOD = 14
STOCHASTIC_PERIOD = 14
WILLIAMS_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
EMA_SPANS = (20, 50, 200)
SWING_WINDOWS = (20, 60)


def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # avg_loss == 0 (no down days at all within the smoothing window --
    # a real possibility on a strong, uninterrupted rally) makes `rs`
    # NaN via the replace() above, which would otherwise sink a
    # legitimately maxed-out RSI to NaN instead of 100. Pegged to 100
    # when there were gains, 50 (flat/undefined) on a truly dead
    # window where avg_gain is also 0.
    warmed_up = avg_gain.notna() & avg_loss.notna()
    no_loss = warmed_up & (avg_loss == 0)
    fallback = pd.Series(np.where(avg_gain > 0, 100.0, 50.0), index=close.index)
    return rsi.where(~no_loss, fallback)


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ADX_PERIOD) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift()
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan))
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def _vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical_price = (high + low + close) / 3
    cumulative_volume = volume.cumsum().replace(0, np.nan)
    return (typical_price * volume).cumsum() / cumulative_volume


def _bollinger(close: pd.Series, period: int = BOLLINGER_PERIOD, num_std: float = BOLLINGER_STD) -> tuple[pd.Series, pd.Series]:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return mid + num_std * std, mid - num_std * std


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = STOCHASTIC_PERIOD) -> tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(period).min()
    highest_high = high.rolling(period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    percent_k = 100 * (close - lowest_low) / denom
    percent_d = percent_k.rolling(3).mean()
    return percent_k, percent_d


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = WILLIAMS_PERIOD) -> pd.Series:
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    return -100 * (highest_high - close) / denom


def _last(series: pd.Series) -> Optional[float]:
    if series.empty:
        return None
    value = series.iloc[-1]
    if pd.isna(value):
        return None
    return round(float(value), 2)


def _series_for_chart(series: pd.Series, valid_dates: set) -> list:
    """[[date_iso, value_or_None], ...] -- NaN (warm-up period before an
    indicator has enough history) becomes None, not a dropped point, so
    the frontend's date axis stays aligned with the price series. Only
    dates in `valid_dates` (price_history's own, post-OHLC-filter dates)
    are included, so a row dropped from price_history for missing OHLC
    doesn't leave this series one point longer and out of alignment."""
    return [
        [idx.date().isoformat(), None if pd.isna(v) else round(float(v), 2)]
        for idx, v in series.items()
        if idx.date().isoformat() in valid_dates
    ]


def build_technicals(ticker: str, range_period: str = "1y") -> dict:
    loader = MarketDataLoader(ticker)
    df = loader.get_historical_prices(period=range_period)

    high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]

    rsi = _rsi(close)
    macd_line, signal_line, histogram = _macd(close)
    adx = _adx(high, low, close)
    vwap = _vwap(high, low, close, volume)
    bollinger_upper, bollinger_lower = _bollinger(close)
    percent_k, percent_d = _stochastic(high, low, close)
    williams_r = _williams_r(high, low, close)
    emas = {span: close.ewm(span=span, adjust=False).mean() for span in EMA_SPANS}
    smas = {span: close.rolling(span).mean() for span in EMA_SPANS}

    last_close = _last(close)
    ema_last = {span: _last(series) for span, series in emas.items()}
    sma_last = {span: _last(series) for span, series in smas.items()}

    # Trend: EMA stack order (20 > 50 > 200 = Bullish, reversed =
    # Bearish) -- the standard, widely-used "EMA ribbon" trend
    # convention, not a proprietary signal. Deliberately does NOT also
    # gate on MACD histogram sign: a real trend's histogram routinely
    # dips through zero on minor pullbacks/pauses without the
    # underlying trend actually reversing, so AND-ing it in would make
    # the label flicker to "Neutral" inside an otherwise clear trend.
    # MACD is still exposed on its own in `indicators` for anyone who
    # wants that separate momentum read.
    trend = "Neutral"
    if ema_last[20] is not None and ema_last[50] is not None and ema_last[200] is not None:
        if ema_last[20] > ema_last[50] > ema_last[200]:
            trend = "Bullish"
        elif ema_last[20] < ema_last[50] < ema_last[200]:
            trend = "Bearish"

    # Moving-average signal: majority vote of close vs each of
    # SMA/EMA 20/50/200 (6 votes) -- documented convention, not a
    # vendor "Strong Buy/Strong Sell" scale this app has no basis for.
    buy_votes = sum(1 for v in list(ema_last.values()) + list(sma_last.values()) if v is not None and last_close is not None and last_close > v)
    sell_votes = sum(1 for v in list(ema_last.values()) + list(sma_last.values()) if v is not None and last_close is not None and last_close < v)
    ma_verdict = "Buy" if buy_votes > sell_votes else "Sell" if sell_votes > buy_votes else "Neutral"

    support_resistance = {}
    for window in SWING_WINDOWS:
        support_resistance[f"support_{window}d"] = _last(low.rolling(window).min())
        support_resistance[f"resistance_{window}d"] = _last(high.rolling(window).max())

    # Rows missing any of O/H/L/C are skipped entirely (a candlestick
    # can't render with a hole in its own body) -- real yfinance
    # history occasionally has one, e.g. an in-progress current trading
    # day fetched mid-session. Volume alone missing is still tolerated
    # (kept as None), matching the cheaper/independent-field convention
    # used everywhere else in this app (get_corporate_actions,
    # financial statement rows).
    price_history = [
        {
            "date": idx.date().isoformat(),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else None,
        }
        for idx, row in df.iterrows()
        if pd.notna(row["Open"]) and pd.notna(row["High"]) and pd.notna(row["Low"]) and pd.notna(row["Close"])
    ]
    # Every overlay is filtered to this same date set -- otherwise a
    # row dropped above for missing OHLC would leave the overlay series
    # one point longer than price_history and out of date-alignment
    # with it (the frontend chart zips them together by position).
    valid_dates = {row["date"] for row in price_history}

    return {
        "price_history": price_history,
        "overlays": {
            "ema_20": _series_for_chart(emas[20], valid_dates),
            "ema_50": _series_for_chart(emas[50], valid_dates),
            "ema_200": _series_for_chart(emas[200], valid_dates),
            "bollinger_upper": _series_for_chart(bollinger_upper, valid_dates),
            "bollinger_lower": _series_for_chart(bollinger_lower, valid_dates),
            "vwap": _series_for_chart(vwap, valid_dates),
        },
        "indicators": {
            "rsi_14": _last(rsi),
            "macd": {"macd": _last(macd_line), "signal": _last(signal_line), "histogram": _last(histogram)},
            "adx_14": _last(adx),
            "vwap": _last(vwap),
            "ema_20": ema_last[20],
            "ema_50": ema_last[50],
            "ema_200": ema_last[200],
            "stochastic": {"k": _last(percent_k), "d": _last(percent_d)},
            "williams_r": _last(williams_r),
        },
        "trend": trend,
        "moving_average_signal": {"buy": buy_votes, "sell": sell_votes, "verdict": ma_verdict},
        "support_resistance": support_resistance,
    }
