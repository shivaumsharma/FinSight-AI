"""Tests for app/analysis/vol_overlay.py, app/reporting/portfolio_risk_overlay.py and GET /v1/portfolio/risk-overlay.
No network: price history is synthetic and the downloader is patched."""

import numpy as np
import pandas as pd
import pytest

from app.analysis import vol_overlay as vo
from app.reporting import portfolio_risk_overlay as pro


def _prices(daily_returns, start=100.0, tickers=("AAA",), days=60):
    idx = pd.bdate_range("2026-01-01", periods=days)
    r = np.resize(np.asarray(daily_returns, float), days)
    path = start * np.cumprod(1 + r)
    return pd.DataFrame({t: path for t in tickers}, index=idx)


# ---------------------------------------------------------------- pure math

def test_realised_vol_of_a_constant_return_series_is_zero_and_exposure_is_undefined():
    r = pd.Series([0.001] * 40)
    assert vo.realised_volatility(r) == pytest.approx(0.0, abs=1e-12)
    assert vo.suggested_exposure(0.0) is None  # no division by zero


def test_realised_vol_matches_hand_computation():
    r = pd.Series([0.01, -0.01] * 20)   # sample std of +-1% alternating, over the last 21 values
    last = r.tail(vo.WINDOW)
    expected = last.std(ddof=1) * np.sqrt(252)
    assert vo.realised_volatility(r) == pytest.approx(expected)


def test_too_little_data_gives_none():
    assert vo.realised_volatility(pd.Series([0.01] * 5)) is None


def test_exposure_is_target_over_vol_and_never_levered():
    assert vo.suggested_exposure(0.30) == pytest.approx(0.5)     # 15% / 30%
    assert vo.suggested_exposure(0.15) == pytest.approx(1.0)
    assert vo.suggested_exposure(0.05) == 1.0                    # calm market: capped at fully invested, no leverage


def test_portfolio_value_uses_quantities_and_drops_rows_with_missing_prices():
    idx = pd.bdate_range("2026-01-01", periods=3)
    prices = pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [np.nan, 20.0, 20.0]}, index=idx)
    v = vo.portfolio_value_series(prices, {"A": 2, "B": 1})
    assert list(v.values) == [2 * 11 + 20, 2 * 12 + 20]          # first day dropped: B not priced yet


def test_unheld_tickers_in_the_download_are_ignored():
    prices = _prices([0.01, -0.01], tickers=("AAA", "ZZZ"))
    v = vo.portfolio_value_series(prices, {"AAA": 3})
    assert v.iloc[0] == pytest.approx(3 * prices["AAA"].iloc[0])


# ---------------------------------------------------------------- build_overlay

def test_high_vol_portfolio_gets_reduced_exposure_and_reports_evidence():
    out = vo.build_overlay(_prices([0.03, -0.03]), {"AAA": 10})
    assert out["available"] is True
    assert out["realised_volatility_pct"] > 15
    assert out["suggested_exposure_pct"] < 100
    assert out["suggested_exposure_pct"] + out["suggested_cash_pct"] == pytest.approx(100.0, abs=0.11)
    assert out["max_drawdown_in_window_pct"] < 0
    assert "not a return forecast" in out["evidence"]["takeaway"]
    assert "advice" in out["disclaimer"]


def test_calm_portfolio_stays_fully_invested():
    out = vo.build_overlay(_prices([0.0005, 0.0006]), {"AAA": 10})
    assert out["suggested_exposure_pct"] == 100.0 and out["suggested_cash_pct"] == 0.0


def test_not_enough_history_degrades_instead_of_raising():
    out = vo.build_overlay(_prices([0.01], days=8), {"AAA": 1})
    assert out["available"] is False and out["reason"]


# ---------------------------------------------------------------- builder (db + downloader patched)

def test_builder_with_no_holdings(monkeypatch):
    monkeypatch.setattr(pro.db, "get_portfolio_holdings", lambda uid: [])
    assert pro.build_risk_overlay("u")["available"] is False


def test_builder_excludes_non_usd_and_lists_them(monkeypatch):
    monkeypatch.setattr(pro.db, "get_portfolio_holdings",
                        lambda uid: [{"ticker": "AAA", "quantity": 5}, {"ticker": "RELIANCE.NS", "quantity": 10}])
    monkeypatch.setattr(pro, "_download_prices", lambda tickers: _prices([0.02, -0.02], tickers=tuple(tickers)))
    out = pro.build_risk_overlay("u")
    assert out["available"] is True
    assert out["excluded_tickers"] == ["RELIANCE.NS"] and out["holdings_used"] == ["AAA"]


def test_builder_only_non_usd(monkeypatch):
    monkeypatch.setattr(pro.db, "get_portfolio_holdings", lambda uid: [{"ticker": "TCS.NS", "quantity": 1}])
    out = pro.build_risk_overlay("u")
    assert out["available"] is False and out["excluded_tickers"] == ["TCS.NS"]


def test_builder_survives_a_failed_price_fetch(monkeypatch):
    monkeypatch.setattr(pro.db, "get_portfolio_holdings", lambda uid: [{"ticker": "AAA", "quantity": 1}])

    def boom(tickers):
        raise RuntimeError("throttled")

    monkeypatch.setattr(pro, "_download_prices", boom)
    out = pro.build_risk_overlay("u")
    assert out["available"] is False and "unavailable" in out["reason"]


# ---------------------------------------------------------------- endpoint

def test_endpoint_requires_auth_and_returns_the_builder_result(monkeypatch):
    from fastapi.testclient import TestClient
    from app.api import auth, main

    client = TestClient(main.app)
    assert client.get("/v1/portfolio/risk-overlay").status_code in (401, 403)

    main.app.dependency_overrides[auth.get_current_user] = lambda: "user-1"
    try:
        monkeypatch.setattr(main, "build_risk_overlay", lambda uid: {"available": False, "reason": f"stub for {uid}"})
        r = client.get("/v1/portfolio/risk-overlay")
        assert r.status_code == 200 and r.json()["reason"] == "stub for user-1"
    finally:
        main.app.dependency_overrides.clear()
