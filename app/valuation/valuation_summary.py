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

        summary = {}

        summary["Intrinsic Value"] = valuation.get(
            "intrinsic_value",
            "Unavailable"
        )

        summary["Enterprise Value"] = valuation.get(
            "enterprise_value",
            "Unavailable"
        )

        summary["Equity Value"] = valuation.get(
            "equity_value",
            "Unavailable"
        )

        summary["Current Price"] = valuation.get(
            "current_price",
            "Unavailable"
        )

        summary["Upside (%)"] = valuation.get(
            "upside_percent",
            "Unavailable"
        )

        summary["WACC"] = valuation.get(
            "wacc",
            "Unavailable"
        )

        summary["Terminal Growth"] = valuation.get(
            "terminal_growth_rate",
            "Unavailable"
        )

        return summary