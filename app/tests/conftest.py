"""Shared test configuration."""

import pytest


@pytest.fixture(autouse=True)
def _hermetic_fundamentals_source(monkeypatch):
    """MarketDataLoader's default FUNDAMENTALS_SOURCE=auto falls back to LIVE SEC EDGAR when a (mocked) yfinance
    statement is empty or raises, which would make unrelated unit tests network-dependent and slow. Tests default to
    yfinance-only; the EDGAR fallback tests (test_edgar_fundamentals.py) set the variable explicitly."""
    monkeypatch.setenv("FUNDAMENTALS_SOURCE", "yfinance")
