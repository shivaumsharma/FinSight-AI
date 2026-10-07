"""Tests for scripts/live_record.py's pure helpers (no network)."""

import numpy as np
import pandas as pd

from scripts import live_record as lr


def _rows(date, tickers, score=10.0):
    return pd.DataFrame({
        "snapshot_date": date, "ticker": tickers, "category": "X", "composite_score": score, "dcf_score": score,
        "relative_score": score, "upside_pct": score, "recommendation": "Hold", "price": 100.0,
    })[lr.COLUMNS]


def test_append_to_empty_record():
    out = lr.append_snapshot(pd.DataFrame(columns=lr.COLUMNS), _rows("2026-10-05", ["B", "A"]))
    assert list(out["ticker"]) == ["A", "B"]


def test_record_is_append_only_same_day_rerun_cannot_rewrite_history():
    first = lr.append_snapshot(pd.DataFrame(columns=lr.COLUMNS), _rows("2026-10-05", ["A", "B"], score=10.0))
    again = lr.append_snapshot(first, _rows("2026-10-05", ["A", "B", "C"], score=99.0))
    assert len(again) == 3
    assert again.set_index("ticker").loc["A", "composite_score"] == 10.0   # original kept, not overwritten
    assert again.set_index("ticker").loc["C", "composite_score"] == 99.0   # genuinely new row added


def test_new_date_is_added_alongside_old():
    first = lr.append_snapshot(pd.DataFrame(columns=lr.COLUMNS), _rows("2026-10-05", ["A"]))
    both = lr.append_snapshot(first, _rows("2026-10-12", ["A"]))
    assert list(both["snapshot_date"]) == ["2026-10-05", "2026-10-12"]


def _prices():
    idx = pd.bdate_range("2026-10-01", "2027-01-29")
    return pd.Series(np.linspace(100, 200, len(idx)), index=idx)


def test_forward_return_uses_first_close_on_or_after_each_end():
    p = _prices()
    r = lr.forward_return(p, "2026-10-05", 63)
    start = p[p.index >= "2026-10-05"].iloc[0]
    end = p[p.index >= pd.Timestamp("2026-10-05") + pd.Timedelta(days=63)].iloc[0]
    assert r == end / start - 1


def test_forward_return_is_none_until_the_horizon_has_matured():
    assert lr.forward_return(_prices(), "2027-01-15", 63) is None
    assert lr.forward_return(pd.Series(dtype=float), "2026-10-05", 63) is None


def test_ic_summary_refuses_a_t_statistic_on_too_few_dates():
    s = lr.ic_summary(pd.Series([0.05, 0.02, -0.01]))
    assert s["t"] is None and "need" in s["note"]
    assert abs(s["mean_ic"] - 0.02) < 1e-9


def test_ic_summary_reports_t_once_enough_dates_matured():
    s = lr.ic_summary(pd.Series([0.05, 0.02, 0.03, 0.04, 0.01, 0.06, 0.02, 0.03, 0.04]))
    assert s["t"] is not None and s["t"] > 0 and s["note"] is None
