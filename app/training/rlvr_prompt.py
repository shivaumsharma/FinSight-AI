"""
rlvr_prompt.py

Direction-prediction prompt for RLVR training -- distinct from
app/reporting/narrative_builder.py's prompt, which hands the model an
already-decided Buy/Hold/Sell rating and asks it to narrate around it.
This prompt asks the model to PRODUCE the call itself.

What's included vs. withheld is the whole point of this file:

- Included: financial summary (revenue/margins/growth/trend), DCF
  intrinsic value and current price (when DCF is available), and the
  relative-valuation multiple comparison -- the same underlying facts a
  human analyst would have, and the same facts narrative_builder.py's
  prompt already cites as context.
- Withheld: the rule-based `rating`/`basis` string, and the
  dcf_score/relative_score/composite_score/upside_percent fields from
  app/reporting/report_data_builder.py's derive_recommendation(). Those
  ARE the deterministic rule's own output -- handing them to the model
  would let it just restate the rule (which already loses to a naive
  always-Buy baseline, see EVALUATION.md) instead of learning anything.
  Current price and intrinsic value are both given as separate raw
  numbers rather than a single pre-computed "Upside: X%" line, so the
  model has to do at least its own synthesis, not copy a graded field.
"""

import re
from typing import Any, Dict, Optional

VALID_VERDICTS = ("Buy", "Hold", "Sell")

_VERDICT_LINE_RE = re.compile(r"Verdict:\s*(Buy|Hold|Sell)\b", re.IGNORECASE)

_INSTRUCTIONS = """You are an equity research analyst. Based ONLY on the facts below, decide \
whether this stock is a Buy, Hold, or Sell.

Give 2-4 sentences of reasoning, then end your response with exactly one line in this \
format (no other text on that line):

Verdict: <Buy, Hold, or Sell>
"""


_DOLLAR_KEYS = {"Revenue", "EBIT", "Net Income", "Free Cash Flow"}


def _fmt_value(key: str, value: Any) -> str:
    """B/M-suffixed formatting for large dollar figures, same shape as
    app/reporting/pdf_report_builder.py's _fmt_number -- kept as its own
    small copy rather than importing that reporting-layer module from
    the training layer, since the two have no other reason to depend on
    each other."""
    if key in _DOLLAR_KEYS and isinstance(value, (int, float)):
        abs_value = abs(value)
        if abs_value >= 1e9:
            return f"${value / 1e9:,.2f}B"
        if abs_value >= 1e6:
            return f"${value / 1e6:,.2f}M"
        return f"${value:,.2f}"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_prompt(ticker: str, financial_summary: Dict[str, Any], valuation_results: Dict[str, Any]) -> str:
    lines = [_INSTRUCTIONS, f"\nCompany: {ticker}\n", "Financials (most recent fiscal year known as of this date):"]

    for key in (
        "Revenue", "EBIT", "Net Income", "Free Cash Flow",
        "Operating Margin", "Net Margin", "ROE", "Debt to Equity",
        "Revenue Growth (%)", "EBIT Growth (%)", "Net Income Growth (%)", "FCF Growth (%)",
        "Revenue Trend", "EBIT Trend", "Net Income Trend", "FCF Trend",
    ):
        value = financial_summary.get(key)
        if value is not None and value != "Unavailable":
            lines.append(f"- {key}: {_fmt_value(key, value)}")

    current_price = valuation_results.get("current_price")
    if current_price is not None:
        lines.append(f"\nCurrent share price: ${current_price:.2f}")

    if valuation_results.get("dcf_available") and valuation_results.get("intrinsic_value") is not None:
        lines.append(f"DCF-modeled intrinsic value per share: ${valuation_results['intrinsic_value']:.2f}")
    else:
        lines.append(
            "DCF valuation: not available for this company "
            f"({valuation_results.get('dcf_unavailable_reason') or 'insufficient data'})."
        )

    relative = valuation_results.get("relative_valuation")
    if relative:
        lines.append(
            f"\nRelative valuation ({relative['metric']}): currently {relative['current_ev_ebitda']:.2f}x "
            f"vs. its own {relative['years_used']}-year historical average of "
            f"{relative['historical_avg_ev_ebitda']:.2f}x ({relative['vs_history_pct']:+.1f}%, "
            f"signal: {relative['signal']})."
        )

    return "\n".join(lines)


def parse_verdict(completion: str) -> Optional[str]:
    """Extracts the Buy/Hold/Sell verdict from a model completion.
    Returns None if the completion didn't follow the requested format --
    callers (rlvr_reward.py) should treat that as an un-scoreable
    rollout, same as phase2_backtest.score_rating() already treats an
    "Insufficient Data" rating as unscored rather than wrong."""
    matches = _VERDICT_LINE_RE.findall(completion or "")
    if not matches:
        return None
    # Last match, not first: a model reasoning out loud may mention
    # "Buy" or "Sell" in prose before its own final "Verdict:" line: the
    # LAST properly-formatted "Verdict:" line is the one actually
    # intended as the answer, matching the format instructed in the
    # prompt (one line, at the end).
    return matches[-1].capitalize()
