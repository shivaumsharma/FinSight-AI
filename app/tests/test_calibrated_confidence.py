"""
Tests for app/reporting/calibrated_confidence.py.
"""

import json

import pytest

from app.reporting import calibrated_confidence as cc


def _row(ticker, category, recommendation, realized_return_pct):
    return {
        "ticker": ticker,
        "category": category,
        "as_of_date": "2025-09-07",
        "recommendation": recommendation,
        "realized_return_pct": realized_return_pct,
    }


def _write_sources(tmp_path, rows_a, rows_b=None):
    (tmp_path / cc._SOURCE_FILES[0]).write_text(json.dumps(rows_a))
    (tmp_path / cc._SOURCE_FILES[1]).write_text(json.dumps(rows_b or []))


# ---------------------------------------------------------------- _sector_from_category

def test_sector_from_category_strips_the_index_suffix():
    assert cc._sector_from_category("Information Technology (S&P 500)") == "Information Technology"


def test_sector_from_category_none_for_missing_category():
    assert cc._sector_from_category(None) is None


# ---------------------------------------------------------------- build_calibrated_confidence

def test_returns_none_for_a_non_call_rating(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "_SCRIPT_DIR", tmp_path)
    assert cc.build_calibrated_confidence("Technology", "Insufficient Data") is None


def test_returns_none_when_no_source_files_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "_SCRIPT_DIR", tmp_path)
    assert cc.build_calibrated_confidence("Technology", "Buy") is None


def test_sector_level_bucket_used_when_it_has_enough_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "_SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(cc, "MIN_SAMPLE_SIZE", 5)
    # 5 Buy calls in Information Technology, 4 correct (>5% realized return)
    rows = [_row(f"T{i}", "Information Technology (S&P 500)", "Buy", 10.0 if i < 4 else 1.0) for i in range(5)]
    _write_sources(tmp_path, rows)

    result = cc.build_calibrated_confidence("Technology", "Buy")

    assert result == {"accuracy_pct": 80.0, "n": 5, "scope": "sector", "sector": "Information Technology", "rating": "Buy"}


def test_falls_back_to_overall_when_sector_bucket_is_too_small(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "_SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(cc, "MIN_SAMPLE_SIZE", 5)
    # Only 2 IT Buy calls (too few) but 5 Buy calls overall across sectors.
    rows = [
        _row("T1", "Information Technology (S&P 500)", "Buy", 10.0),
        _row("T2", "Information Technology (S&P 500)", "Buy", 10.0),
        _row("T3", "Financials (S&P 500)", "Buy", 10.0),
        _row("T4", "Financials (S&P 500)", "Buy", 10.0),
        _row("T5", "Energy (S&P 500)", "Buy", -10.0),
    ]
    _write_sources(tmp_path, rows)

    result = cc.build_calibrated_confidence("Technology", "Buy")

    assert result["scope"] == "overall"
    assert result["sector"] is None
    assert result["n"] == 5


def test_returns_none_when_even_the_overall_bucket_is_too_small(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "_SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(cc, "MIN_SAMPLE_SIZE", 20)
    rows = [_row("T1", "Information Technology (S&P 500)", "Buy", 10.0)]
    _write_sources(tmp_path, rows)

    assert cc.build_calibrated_confidence("Technology", "Buy") is None


def test_regression_yfinance_sector_naming_must_be_mapped_to_gics(tmp_path, monkeypatch):
    # Regression test for the real bug caught while building this:
    # yfinance's own sector taxonomy ("Technology", "Consumer Cyclical",
    # ...) does NOT match the backtest's GICS-style category field
    # ("Information Technology", "Consumer Discretionary", ...) --
    # without _YFINANCE_SECTOR_TO_GICS, a real ticker's sector would
    # NEVER match sector-level data at all, silently degrading every
    # report to the less-specific "overall" bucket.
    monkeypatch.setattr(cc, "_SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(cc, "MIN_SAMPLE_SIZE", 5)
    rows = [_row(f"T{i}", "Information Technology (S&P 500)", "Buy", 10.0) for i in range(5)]
    _write_sources(tmp_path, rows)

    # "Technology" is yfinance's real sector string for a company GICS
    # classifies as "Information Technology".
    result = cc.build_calibrated_confidence("Technology", "Buy")

    assert result is not None
    assert result["scope"] == "sector"
    assert result["sector"] == "Information Technology"


def test_unmapped_sector_falls_back_to_overall_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(cc, "_SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(cc, "MIN_SAMPLE_SIZE", 3)
    rows = [_row(f"T{i}", "Energy (S&P 500)", "Buy", 10.0) for i in range(3)]
    _write_sources(tmp_path, rows)

    result = cc.build_calibrated_confidence("Some Made Up Sector", "Buy")

    assert result["scope"] == "overall"
