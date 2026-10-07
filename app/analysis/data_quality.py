"""
data_quality.py

How complete the inputs behind one report are, shown beside the rating. Display-only: it never changes the
recommendation. `no_view` flags reports with too little input data. It is a disclosure, not an accuracy filter:
the pre-registered test (SPRINT_TRACKER.md, "FIX 1") found no accuracy difference by completeness.
"""

from typing import Any, Dict, Optional

KEY_FIELDS = ["Revenue", "EBIT", "Net Income", "Free Cash Flow", "Operating Margin", "ROE", "EPS",
              "Debt to Equity", "Revenue CAGR (%)"]
HIGH_THRESHOLD = 85
LOW_THRESHOLD = 60


def _present(value: Any) -> bool:
    return value is not None and value != "Unavailable" and value != "Unknown"


def build_data_quality(financial: Optional[Dict[str, Any]], valuation_results: Optional[Dict[str, Any]],
                       company_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    financial, valuation_results, company_info = financial or {}, valuation_results or {}, company_info or {}
    missing = [f for f in KEY_FIELDS if not _present(financial.get(f))]
    completeness = 100 * (len(KEY_FIELDS) - len(missing)) / len(KEY_FIELDS)

    dcf_ok = bool(valuation_results.get("dcf_available", True))
    has_price = _present(company_info.get("current_price"))
    has_relative = bool(valuation_results.get("relative_valuation"))

    score = 0.6 * completeness + 25 * dcf_ok + 10 * has_price + 5 * has_relative
    issues = []
    if missing:
        issues.append(f"{len(missing)} of {len(KEY_FIELDS)} key financial fields missing: {', '.join(missing)}")
    if not dcf_ok:
        issues.append("DCF unavailable" + (f" ({valuation_results['dcf_unavailable_reason']})"
                                           if valuation_results.get("dcf_unavailable_reason") else ""))
    if not has_price:
        issues.append("no current price")
    if not has_relative:
        issues.append("no peer-relative valuation")

    level = "HIGH" if score >= HIGH_THRESHOLD else "LOW" if score < LOW_THRESHOLD else "MEDIUM"
    return {"score": round(score), "level": level, "no_view": level == "LOW" or not dcf_ok,
            "issues": issues, "fields_present": len(KEY_FIELDS) - len(missing), "fields_total": len(KEY_FIELDS)}
