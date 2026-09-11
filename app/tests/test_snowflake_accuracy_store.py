"""
Unit tests for scripts/snowflake_accuracy_store.py -- no real Snowflake
account needed. connect()'s no-op-when-unconfigured contract and the
raw-row-to-table-row mapping are pure/deterministic; write_rows/
run_validation_queries are exercised against a fake cursor/connection
object instead of a real snowflake.connector.Connection.
"""

import pytest

from scripts import snowflake_accuracy_store as sas


def test_connect_returns_none_when_no_env_vars_are_set(monkeypatch):
    for var in ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD", "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE"):
        monkeypatch.delenv(var, raising=False)
    assert sas.connect() is None


def test_connect_returns_none_when_only_partially_configured(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct")
    monkeypatch.setenv("SNOWFLAKE_USER", "user")
    for var in ("SNOWFLAKE_PASSWORD", "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE"):
        monkeypatch.delenv(var, raising=False)
    assert sas.connect() is None


def test_row_to_record_maps_raw_backtest_fields_to_table_columns():
    raw_row = {
        "ticker": "MMM",
        "category": "Industrials (S&P 500)",
        "as_of_date": "2025-09-07",
        "recommendation": "Hold",
        "realized_return_pct": 11.64,
        "correct": False,
    }
    record = sas._row_to_record(raw_row, source_file="backtest_results_a.json")
    assert record == {
        "ticker": "MMM",
        "call": "Hold",
        "actual_return_12m": 11.64,
        "correct": False,
        "run_date": "2025-09-07",
        "universe": "Industrials (S&P 500)",
        "model_version": sas.MODEL_VERSION,
        "source_file": "backtest_results_a.json",
    }


def test_row_to_record_coerces_missing_correct_to_false():
    # A row with no "correct" key at all (shouldn't happen for a
    # scoreable row, but write_rows must never raise KeyError on it).
    raw_row = {"ticker": "AAPL", "recommendation": "Buy", "realized_return_pct": 5.0}
    record = sas._row_to_record(raw_row, source_file="x.json")
    assert record["correct"] is False


class _FakeCursor:
    def __init__(self, executed):
        self._executed = executed

    def execute(self, sql):
        self._executed.append(sql)

    def fetchall(self):
        return [("Technology", 100, 60, 60.0)]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeConnection:
    def __init__(self):
        self.executed = []

    def cursor(self):
        return _FakeCursor(self.executed)


def test_ensure_table_executes_create_table_sql():
    conn = _FakeConnection()
    sas.ensure_table(conn)
    assert len(conn.executed) == 1
    assert sas.TABLE_NAME in conn.executed[0]
    assert "CREATE TABLE IF NOT EXISTS" in conn.executed[0]


def test_write_rows_returns_zero_for_empty_input():
    conn = _FakeConnection()
    assert sas.write_rows(conn, {}) == 0
    assert sas.write_rows(conn, {"f.json": []}) == 0


def test_run_validation_queries_executes_all_three_named_queries():
    conn = _FakeConnection()
    results = sas.run_validation_queries(conn)
    assert set(results.keys()) == set(sas.VALIDATION_QUERIES.keys())
    assert len(conn.executed) == 3
    for name in sas.VALIDATION_QUERIES:
        assert sas.TABLE_NAME in sas.VALIDATION_QUERIES[name]
