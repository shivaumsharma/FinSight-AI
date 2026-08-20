"""
Unit tests for app/training/rlvr_context.py's build_point_in_time_context
-- specifically the NaN-price guard. Found live: a full dataset build
(scripts/build_rlvr_dataset.py) silently produced realized_return_pct
= NaN for every single example, because yfinance's own Close column
had a NaN row on-or-before the target date, and the code only guarded
against `None` (missing entirely), not a real NaN (present but bad).
Downstream, score_rating(verdict, nan) returns False (not None), so
GRPO's reward was silently guaranteed 0.0 for an entire training run
regardless of what the model predicted -- this test is what should
have caught that before it cost four hours of GPU time.

Mocks only the yfinance.Ticker boundary (this module's own real
network dependency), not build_point_in_time_context's internals --
the NaN guard fires before any financial-statement or ValuationTool
code ever runs, so nothing else needs mocking for this path.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.training.rlvr_context import PointInTimeUnavailable, build_point_in_time_context


def _fake_ticker(closes: dict):
    """closes: {date_str: close_price}: build a fake yfinance.Ticker
    whose .history() returns exactly these rows, NaN values included."""
    index = pd.to_datetime(list(closes.keys()))
    df = pd.DataFrame({"Close": list(closes.values())}, index=index)

    class _FakeTicker:
        def history(self, period):
            return df

    return _FakeTicker()


def test_nan_close_on_or_before_as_of_date_raises_not_silently_nan():
    as_of_date = pd.Timestamp("2024-01-15")
    today_date = pd.Timestamp("2025-01-15")
    fake = _fake_ticker({
        "2024-01-10": np.nan,  # last real row on-or-before as_of_date -- NaN, not missing
        "2025-01-10": 150.0,
    })

    with patch("app.training.rlvr_context.yf.Ticker", return_value=fake):
        with pytest.raises(PointInTimeUnavailable, match="NaN price"):
            build_point_in_time_context("FAKE", as_of_date, today_date, market_history=None)


def test_nan_close_on_or_before_today_raises_not_silently_nan():
    as_of_date = pd.Timestamp("2024-01-15")
    today_date = pd.Timestamp("2025-01-15")
    fake = _fake_ticker({
        "2024-01-10": 100.0,
        "2025-01-10": np.nan,  # last real row on-or-before today_date -- NaN
    })

    with patch("app.training.rlvr_context.yf.Ticker", return_value=fake):
        with pytest.raises(PointInTimeUnavailable, match="NaN price"):
            build_point_in_time_context("FAKE", as_of_date, today_date, market_history=None)


def test_valid_prices_do_not_trip_the_nan_guard():
    # Confirms the guard is specific to NaN, not an overly broad check
    # that would reject perfectly good data -- the fake ticker only
    # stubs .history(), so real (non-NaN) prices should sail past the
    # guard and fail LATER, on the next real step (fetching financial
    # statements, which isn't stubbed here), never on "NaN price".
    as_of_date = pd.Timestamp("2024-01-15")
    today_date = pd.Timestamp("2025-01-15")
    fake = _fake_ticker({
        "2024-01-10": 100.0,
        "2025-01-10": 150.0,
    })

    with patch("app.training.rlvr_context.yf.Ticker", return_value=fake):
        with pytest.raises(Exception) as exc_info:
            build_point_in_time_context("FAKE", as_of_date, today_date, market_history=None)
    assert "NaN price" not in str(exc_info.value)
