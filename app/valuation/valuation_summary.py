"""
Creates a concise valuation summary
from the DCF pipeline.
"""

from typing import Dict


class ValuationSummaryBuilder:

    def build(
        self,
        valuation: Dict
    ) -> Dict:

        if not valuation:
            return {}

        def _or_unavailable(key):
            # dict.get(key, "Unavailable") only falls back when the key
            # is absent -- it does NOT catch a key that's present with
            # value None (which is exactly what the DCF-unavailable
            # path in ValuationPipeline sets for every numeric field).
            # Left as plain .get(), "None" would leak into the report
            # and PDF as literal text instead of "Unavailable".
            value = valuation.get(key)
            return value if value is not None else "Unavailable"

        summary = {}

        summary["Intrinsic Value"] = _or_unavailable("intrinsic_value")
        summary["Enterprise Value"] = _or_unavailable("enterprise_value")
        summary["Equity Value"] = _or_unavailable("equity_value")
        summary["Current Price"] = _or_unavailable("current_price")
        summary["Upside (%)"] = _or_unavailable("upside_percent")
        summary["WACC"] = _or_unavailable("wacc")
        summary["Raw WACC"] = _or_unavailable("raw_wacc")
        summary["WACC Floored"] = valuation.get("wacc_floored", False)
        summary["WACC Floor Note"] = valuation.get("wacc_floor_note")
        summary["Terminal Growth"] = _or_unavailable("terminal_growth_rate")

        return summary