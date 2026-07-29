"""
serialization.py

Converts a fully-populated ResearchContext into a JSON-safe dict for
the API response. Single source of truth for this shape -- used by
jobs.py when a job completes, and by app/tests/test_api.py.

The DataFrame problem
----------------------
context.report_data["valuation_analysis"]["sensitivity_table"] (see
report_data_builder.py) is a raw pandas DataFrame -- FastAPI's default
encoder can't serialize it as-is. Checked directly (not assumed):
streamlit_app.py never reads it (`grep -n "sensitivity" streamlit_app.py`
returns nothing) -- only pdf_report_builder.py does, and that already
runs server-side inside report_tool.py before this function is ever
called, producing plain PDF bytes. So it's KEPT here (converted via
.to_dict(), not dropped) purely because a future client legitimately
might want to show it, not because anything currently reads it.

context.valuation_results["fcff_forecasts"] is a second, separate
DataFrame that genuinely is dropped -- nothing (not even the PDF
builder) reads it today, and this function doesn't include raw
valuation_results at all, only report_data plus the one thing a client
needs that report_data doesn't carry: normalized_financials, needed by
the What-If DCF sliders (FCFFEngine(financial_df)... in streamlit_app.py,
once that's rewired to call this API in step 4).

normalized_financials round-trips via to_json(orient="split") /
read_json(orient="split") -- the orientation that preserves both the
datetime index and column order that .iloc[-1]/.dropna() calls
elsewhere in the codebase depend on. See
app/tests/test_api.py::test_normalized_financials_round_trip for the
dedicated correctness check on this specific conversion -- a silent
dtype/index mismatch here would make the DCF quietly compute different
numbers with no visible error, which is exactly the kind of failure
that needs its own test rather than being inferred from a passing API test.
"""

import copy
import io
from typing import Any, Dict, Optional

import pandas as pd

from app.core.research_context import ResearchContext


def context_to_api_dict(context: ResearchContext) -> Dict[str, Any]:
    report_data = copy.deepcopy(context.report_data) if context.report_data else {}

    sensitivity_table = report_data.get("valuation_analysis", {}).get("sensitivity_table")
    if sensitivity_table is not None and hasattr(sensitivity_table, "to_dict"):
        report_data["valuation_analysis"]["sensitivity_table"] = sensitivity_table.to_dict()

    normalized_financials_json = None
    if context.normalized_financials is not None and not context.normalized_financials.empty:
        normalized_financials_json = context.normalized_financials.to_json(orient="split", date_format="iso")

    return {
        "ticker": context.ticker,
        "mode": context.mode,
        "peer_ticker": context.metadata.get("peer_ticker"),
        "report_data": report_data,
        "normalized_financials": normalized_financials_json,
        "tool_trace": context.tool_trace,
    }


def financial_df_from_json(raw_json: str) -> Optional[pd.DataFrame]:
    """
    Reverses the to_json(orient="split") conversion above. Deliberately
    lives next to context_to_api_dict rather than in whatever eventually
    consumes it (a rewired streamlit_app.py, a future mobile client) --
    single source of truth for both directions of this specific
    conversion, so they can't drift out of sync with each other.

    pd.read_json() (pandas 3.x, confirmed directly against this venv's
    installed version) requires a file-like object, not a raw string --
    passing the JSON string directly raises FileNotFoundError because
    it's interpreted as a path. io.StringIO wraps it correctly.
    """
    if raw_json is None:
        return None
    return pd.read_json(io.StringIO(raw_json), orient="split")
