"""Tests for app/analysis/data_quality.py and app/analysis/risk_range.py."""

import math

import numpy as np
import pandas as pd

from app.analysis import data_quality as dq
from app.analysis import risk_range as rr


def _full_financials():
    return {f: 1.0 for f in dq.KEY_FIELDS}


# ---------------------------------------------------------------- data quality

def test_complete_inputs_score_high_and_have_a_view():
    out = dq.build_data_quality(_full_financials(), {"dcf_available": True, "relative_valuation": {"signal": "x"}},
                                {"current_price": 100})
    assert out["level"] == "HIGH" and out["score"] == 100 and not out["no_view"] and out["issues"] == []


def test_missing_fields_are_listed_and_lower_the_score():
    fin = _full_financials()
    fin["ROE"], fin["EPS"] = "Unavailable", None
    out = dq.build_data_quality(fin, {"dcf_available": True, "relative_valuation": {"x": 1}}, {"current_price": 100})
    assert out["fields_present"] == len(dq.KEY_FIELDS) - 2
    assert "ROE" in out["issues"][0] and "EPS" in out["issues"][0]
    assert out["score"] < 100


def test_unavailable_dcf_means_no_view_even_when_the_rest_is_complete():
    out = dq.build_data_quality(_full_financials(), {"dcf_available": False, "dcf_unavailable_reason": "negative FCF"},
                                {"current_price": 100})
    assert out["no_view"] is True
    assert any("DCF unavailable" in i and "negative FCF" in i for i in out["issues"])


def test_empty_inputs_are_low_and_do_not_crash():
    out = dq.build_data_quality(None, None, None)
    assert out["level"] == "LOW" and out["no_view"] is True


# ---------------------------------------------------------------- risk range

def _prices(daily_vol, n=300, start=100.0, seed=0):
    r = np.random.default_rng(seed).normal(0, daily_vol, n)
    return pd.DataFrame({"Close": start * np.exp(np.cumsum(r))})


def test_price_range_matches_the_lognormal_formula():
    out = rr.price_range(100.0, 0.20, 63)
    sd = 0.20 * math.sqrt(63 / 252)
    assert out["high"] == round(100 * math.exp(rr.Z_80 * sd), 2)
    assert out["low"] == round(100 * math.exp(-rr.Z_80 * sd), 2)
    assert out["low"] < 100 < out["high"]


def test_longer_horizon_and_higher_vol_widen_the_range():
    quiet, wild = rr.price_range(100, 0.15, 63), rr.price_range(100, 0.60, 63)
    assert wild["high"] - wild["low"] > quiet["high"] - quiet["low"]
    year = rr.price_range(100, 0.30, 252)
    quarter = rr.price_range(100, 0.30, 63)
    assert year["high"] > quarter["high"] and year["low"] < quarter["low"]


def test_build_risk_range_estimates_volatility_and_uses_last_price():
    prices = _prices(0.02, n=400)
    out = rr.build_risk_range(prices)
    assert 25 < out["annualised_volatility_pct"] < 40  # 0.02 daily is about 32% annualised
    assert out["current_price"] == round(float(prices["Close"].iloc[-1]), 2)
    assert set(out["ranges"]) == {"1 quarter", "1 year"}


def test_build_risk_range_is_none_without_enough_history_or_a_close_column():
    assert rr.build_risk_range(None) is None
    assert rr.build_risk_range(pd.DataFrame()) is None
    assert rr.build_risk_range(_prices(0.02, n=50)) is None
    assert rr.build_risk_range(pd.DataFrame({"Open": [1.0] * 300})) is None


# ---------------------------------------------------------------- prediction log: challenger + durable mirror

def _report_data():
    return {
        "recommendation": {"rating": "Buy"},
        "valuation_analysis": {"DCF Available": True, "Upside (%)": 12.0, "relative_valuation": {"signal": "Undervalued"},
                               "ml_classifier": {"verdict": "OVERVALUED"}},
        "confidence_scores": {"Grounding (%)": 80, "Overall Score": 75},
        "market_earnings_snapshot": {"current_price": 101.5},
        "data_quality": {"level": "HIGH"},
        "risk_range": {"ranges": {"1 quarter": {"low": 90.0, "high": 112.0}}},
    }


def test_log_records_price_challenger_quality_and_range(tmp_path, monkeypatch):
    from app.core import prediction_log as pl

    monkeypatch.setattr(pl.PredictionLogger, "_mirror_to_snowflake", staticmethod(lambda e: False))
    entry = pl.PredictionLogger(str(tmp_path / "log.jsonl")).log("AAPL", _report_data())
    assert entry["price_at_call"] == 101.5 and entry["challenger_verdict"] == "OVERVALUED"
    assert entry["data_quality_level"] == "HIGH" and (entry["range_low_1q"], entry["range_high_1q"]) == (90.0, 112.0)
    assert pl.PredictionLogger(str(tmp_path / "log.jsonl")).read_all()[0]["ticker"] == "AAPL"


def test_mirror_is_a_noop_without_snowflake_credentials(monkeypatch):
    from app.core import prediction_log as pl

    for var in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD", "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE"):
        monkeypatch.delenv(var, raising=False)
    assert pl.PredictionLogger._mirror_to_snowflake({"ticker": "AAPL"}) is False


def test_mirror_inserts_one_row_with_every_column(monkeypatch):
    from app.core import prediction_log as pl
    from scripts import snowflake_accuracy_store as store

    executed = []

    class Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql, params=None): executed.append((sql, params))

    class Conn:
        closed = False
        def cursor(self): return Cur()
        def close(self): self.closed = True

    conn = Conn()
    monkeypatch.setattr(store, "connect", lambda: conn)
    entry = {"ticker": "AAPL", "recommendation": "Buy", "price_at_call": 101.5}
    assert pl.PredictionLogger._mirror_to_snowflake(entry) is True
    sql, params = executed[-1]
    assert sql.startswith("INSERT INTO PREDICTION_LOG") and len(params) == len(pl._SNOWFLAKE_COLUMNS)
    assert params[pl._SNOWFLAKE_COLUMNS.index("ticker")] == "AAPL" and conn.closed
