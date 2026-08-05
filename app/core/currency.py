"""
currency.py

Single shared currency-symbol lookup, used by every layer that
formats a monetary value (app/valuation/valuation_pipeline.py,
app/reporting/report_data_builder.py, app/reporting/pdf_report_builder.py,
and surfaced to web/ via report_data's top-level "currency_symbol"
field). Lives in app/core/ rather than app/reporting/ so app/valuation/
can depend on it without creating a valuation -> reporting import
(the wrong direction -- reporting already depends on valuation output).
"""

CURRENCY_SYMBOLS = {"USD": "$", "INR": "₹"}
DEFAULT_CURRENCY = "USD"


def currency_symbol(currency: str) -> str:
    return CURRENCY_SYMBOLS.get(currency or DEFAULT_CURRENCY, CURRENCY_SYMBOLS[DEFAULT_CURRENCY])
