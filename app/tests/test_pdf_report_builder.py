"""
Unit tests for the pure formatter helpers in
app/reporting/pdf_report_builder.py -- zero prior coverage for this
module. Scoped to _fmt_number/_fmt_percent/_alpha_factor_value (real,
isolated logic); build_pdf_report itself is reportlab wiring around
these (a large report_data fixture for a rendering smoke test isn't
worth its own weight here -- the risk lives in the formatting logic,
not in constructing Paragraph/Table objects).
"""

import pytest

from app.reporting.pdf_report_builder import _alpha_factor_value, _fmt_number, _fmt_percent


# ---------------------------------------------------------------- _fmt_number

def test_fmt_number_scales_billions():
    assert _fmt_number(2_500_000_000) == "2.50B"


def test_fmt_number_scales_millions():
    assert _fmt_number(45_000_000) == "45.00M"


def test_fmt_number_leaves_small_values_unscaled():
    assert _fmt_number(1234.5) == "1,234.50"


def test_fmt_number_applies_prefix_and_suffix():
    assert _fmt_number(1234.5, prefix="$") == "$1,234.50"
    assert _fmt_number(45_000_000, prefix="$") == "$45.00M"


def test_fmt_number_negative_values_scale_by_magnitude():
    assert _fmt_number(-2_500_000_000) == "-2.50B"


def test_fmt_number_none_and_sentinel_strings_are_unavailable():
    assert _fmt_number(None) == "Unavailable"
    assert _fmt_number("Unavailable") == "Unavailable"
    assert _fmt_number("Unknown") == "Unavailable"


def test_fmt_number_passes_through_other_strings_unchanged():
    assert _fmt_number("N/A") == "N/A"


# ---------------------------------------------------------------- _fmt_percent

def test_fmt_percent_formats_to_two_decimals():
    assert _fmt_percent(12.3456) == "12.35%"


def test_fmt_percent_none_and_sentinel_are_unavailable():
    assert _fmt_percent(None) == "Unavailable"
    assert _fmt_percent("Unavailable") == "Unavailable"


def test_fmt_percent_passes_through_other_strings_unchanged():
    assert _fmt_percent("N/A") == "N/A"


# ---------------------------------------------------------------- _alpha_factor_value

def test_alpha_factor_value_none_is_unavailable():
    assert _alpha_factor_value(None) == "Unavailable"


def test_alpha_factor_value_formats_a_pe_vs_history_dict():
    value = {
        "current": 28.5,
        "historical_avg": 22.1,
        "years_used": 5,
        "vs_history_pct": 29.0,
        "signal": "expensive",
    }
    assert _alpha_factor_value(value) == "28.5x vs 22.1x 5yr avg (+29.0%, expensive)"


def test_alpha_factor_value_falls_back_to_str_for_an_incomplete_dict():
    # No vs_history_pct/signal -- not the P/E-vs-history shape, so it
    # must not crash trying to format a %+.1f on a missing key.
    value = {"foo": "bar"}
    assert _alpha_factor_value(value) == str(value)


def test_alpha_factor_value_bool_is_not_treated_as_an_int():
    # bool is an int subclass in Python -- must be checked first, or
    # True/False would render as "1"/"0" via the int branch instead.
    assert _alpha_factor_value(True) == "True"
    assert _alpha_factor_value(False) == "False"


def test_alpha_factor_value_formats_int_with_thousands_separator():
    assert _alpha_factor_value(1234567) == "1,234,567"


def test_alpha_factor_value_formats_float_to_two_decimals():
    assert _alpha_factor_value(3.14159) == "3.14"


def test_alpha_factor_value_passes_through_a_plain_string():
    assert _alpha_factor_value("Buy") == "Buy"
