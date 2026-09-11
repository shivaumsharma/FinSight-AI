"""
snowflake_accuracy_store.py

Optional Snowflake storage target for canonical_accuracy.py's per-call
backtest rows -- NOT a persistence-layer migration. SQLite (app/api/db.py)
keeps owning everything else in this app; this module's only job is
giving the canonical-accuracy artifact a queryable home for the kind of
ad-hoc analysis (precision by sector, Buy-vs-Sell precision, accuracy
trend over time) that a single aggregate JSON file can't answer.

Connection is entirely env-var driven, same "absent means degrade to a
no-op, never crash the caller" contract app/core/cache.py's Redis client
already uses for an optional dependency: SNOWFLAKE_ACCOUNT, _USER,
_PASSWORD, _WAREHOUSE, _DATABASE are required; _SCHEMA defaults to
PUBLIC. connect() returns None (not an exception) if any required var
is missing, so canonical_accuracy.py can call this unconditionally and
every caller without Snowflake configured just keeps writing the JSON
file exactly as before.

Auth is plain user/password (SNOWFLAKE_PASSWORD) -- the simplest option
the snowflake-connector-python library supports natively with zero
extra dependencies. Key-pair auth is a reasonable upgrade later if
password rotation becomes a real operational need; not built here
since nothing in this project's scope asked for it yet.
"""

import os
from typing import Any, Dict, List, Optional

TABLE_NAME = "ACCURACY_BACKTEST_CALLS"

_CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    ticker STRING NOT NULL,
    call STRING NOT NULL,
    actual_return_12m FLOAT,
    correct BOOLEAN,
    run_date DATE,
    universe STRING,
    model_version STRING NOT NULL,
    source_file STRING NOT NULL,
    loaded_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
)
"""

# Bumped by hand when derive_recommendation's scoring methodology
# changes meaningfully (e.g. a new composite-score weighting) -- lets
# a validation query separate "the model got worse" from "we're
# comparing two different models." See app/reporting/report_data_builder.py's
# derive_recommendation for what this version actually describes today.
MODEL_VERSION = "dcf_relative_composite_v1"

# Named, not inline, so canonical_accuracy.py and any ad-hoc caller
# (e.g. a notebook) share the exact same three questions instead of
# quietly drifting into slightly different SQL over time.
VALIDATION_QUERIES = {
    "precision_by_sector": f"""
        SELECT
            universe,
            COUNT(*) AS n_calls,
            SUM(IFF(correct, 1, 0)) AS n_correct,
            ROUND(100.0 * SUM(IFF(correct, 1, 0)) / COUNT(*), 1) AS precision_pct
        FROM {TABLE_NAME}
        GROUP BY universe
        ORDER BY n_calls DESC
    """,
    "buy_vs_sell_precision": f"""
        SELECT
            call,
            COUNT(*) AS n_calls,
            SUM(IFF(correct, 1, 0)) AS n_correct,
            ROUND(100.0 * SUM(IFF(correct, 1, 0)) / COUNT(*), 1) AS precision_pct
        FROM {TABLE_NAME}
        WHERE call IN ('Buy', 'Sell')
        GROUP BY call
        ORDER BY call
    """,
    "accuracy_trend_by_run_date": f"""
        SELECT
            run_date,
            COUNT(*) AS n_calls,
            ROUND(100.0 * SUM(IFF(correct, 1, 0)) / COUNT(*), 1) AS accuracy_pct
        FROM {TABLE_NAME}
        GROUP BY run_date
        ORDER BY run_date
    """,
}


def connect():
    """Returns a real snowflake.connector.Connection, or None if the
    required env vars aren't set -- never raises for a missing/absent
    configuration, only for a genuinely broken one (bad credentials
    against a configured account, e.g.), so a caller can safely treat
    None as "not configured, skip this" without a try/except of its own.
    """
    account = os.environ.get("SNOWFLAKE_ACCOUNT")
    user = os.environ.get("SNOWFLAKE_USER")
    password = os.environ.get("SNOWFLAKE_PASSWORD")
    warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE")
    database = os.environ.get("SNOWFLAKE_DATABASE")
    if not all([account, user, password, warehouse, database]):
        return None

    import snowflake.connector

    return snowflake.connector.connect(
        account=account,
        user=user,
        password=password,
        warehouse=warehouse,
        database=database,
        schema=os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC"),
    )


def ensure_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(_CREATE_TABLE_SQL)


def _row_to_record(row: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    return {
        "ticker": row["ticker"],
        "call": row["recommendation"],
        "actual_return_12m": row.get("realized_return_pct"),
        "correct": bool(row.get("correct")),
        "run_date": row.get("as_of_date"),
        "universe": row.get("category"),
        "model_version": MODEL_VERSION,
        "source_file": source_file,
    }


def write_rows(conn, rows_by_source: Dict[str, List[Dict[str, Any]]]) -> int:
    """rows_by_source: {source_filename: [raw backtest row, ...]} -- the
    same shape canonical_accuracy.py's load_source() already produces
    per file, kept separate (rather than pre-flattened) so source_file
    stays accurate per row without the caller having to stamp it on
    each dict itself.

    Uses write_pandas (bulk COPY INTO under the hood, not row-by-row
    INSERTs) since pandas is already a core dependency of this project
    -- no new dependency beyond the Snowflake connector itself. Table
    must already exist (see ensure_table); auto_create_table is
    deliberately off so the schema stays the explicit one in
    _CREATE_TABLE_SQL, not whatever types pandas happens to infer.
    """
    records = [
        _row_to_record(row, source_file)
        for source_file, rows in rows_by_source.items()
        for row in rows
    ]
    if not records:
        return 0

    import pandas as pd
    from snowflake.connector.pandas_tools import write_pandas

    df = pd.DataFrame.from_records(records)
    df.columns = [c.upper() for c in df.columns]
    success, _, n_rows, _ = write_pandas(conn, df, TABLE_NAME)
    if not success:
        raise RuntimeError(f"write_pandas reported failure writing to {TABLE_NAME}")
    return n_rows


def run_validation_queries(conn) -> Dict[str, List[tuple]]:
    results = {}
    with conn.cursor() as cur:
        for name, sql in VALIDATION_QUERIES.items():
            cur.execute(sql)
            results[name] = cur.fetchall()
    return results


def print_validation_report(conn) -> None:
    results = run_validation_queries(conn)
    for name, rows in results.items():
        print(f"\n-- {name} --")
        for row in rows:
            print(f"  {row}")


if __name__ == "__main__":
    conn = connect()
    if conn is None:
        print(
            "Snowflake not configured -- set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, "
            "SNOWFLAKE_PASSWORD, SNOWFLAKE_WAREHOUSE, and SNOWFLAKE_DATABASE "
            "to run validation queries."
        )
        raise SystemExit(1)
    try:
        print_validation_report(conn)
    finally:
        conn.close()
