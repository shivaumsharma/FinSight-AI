"""
pdf_report_builder.py

Renders the report assembled by report_data_builder.py +
narrative_builder.py into a downloadable PDF using reportlab (pure
Python, no native OS dependencies -- important for a clean install on
Windows, unlike e.g. weasyprint's GTK/Cairo requirement).
"""

import io
from datetime import datetime, timezone
from typing import Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)

_STYLES = getSampleStyleSheet()

_TITLE_STYLE = ParagraphStyle(
    "ReportTitle", parent=_STYLES["Title"], fontSize=20, spaceAfter=4,
)
_SUBTITLE_STYLE = ParagraphStyle(
    "ReportSubtitle", parent=_STYLES["Normal"], fontSize=11,
    textColor=colors.grey, spaceAfter=20,
)
_SECTION_STYLE = ParagraphStyle(
    "SectionHeading", parent=_STYLES["Heading2"], fontSize=14,
    spaceBefore=18, spaceAfter=8, textColor=colors.HexColor("#1a1a2e"),
)
_BODY_STYLE = ParagraphStyle(
    "Body", parent=_STYLES["Normal"], fontSize=10, leading=14, spaceAfter=8,
)
_RECOMMENDATION_STYLE = ParagraphStyle(
    "Recommendation", parent=_STYLES["Normal"], fontSize=16,
    spaceAfter=6, spaceBefore=4,
)
_DISCLAIMER_STYLE = ParagraphStyle(
    "Disclaimer", parent=_STYLES["Normal"], fontSize=9.5, leading=13,
    spaceAfter=14, spaceBefore=4, borderColor=colors.HexColor("#9a6700"),
    borderWidth=1, borderPadding=8, backColor=colors.HexColor("#fff8e6"),
)

_RATING_COLORS = {
    "Buy": colors.HexColor("#1a7f37"),
    "Hold": colors.HexColor("#9a6700"),
    "Sell": colors.HexColor("#cf222e"),
    "Insufficient Data": colors.grey,
}


def _fmt_number(value: Any, prefix: str = "", suffix: str = "") -> str:
    if value is None or value == "Unavailable" or value == "Unknown":
        return "Unavailable"
    if isinstance(value, (int, float)):
        abs_value = abs(value)
        if abs_value >= 1e9:
            return f"{prefix}{value / 1e9:,.2f}B{suffix}"
        if abs_value >= 1e6:
            return f"{prefix}{value / 1e6:,.2f}M{suffix}"
        return f"{prefix}{value:,.2f}{suffix}"
    return str(value)


def _fmt_percent(value: Any) -> str:
    if value is None or value == "Unavailable":
        return "Unavailable"
    if isinstance(value, (int, float)):
        return f"{value:.2f}%"
    return str(value)


def _dict_table(data: dict, formatters: Optional[dict] = None) -> Table:
    formatters = formatters or {}
    rows = []
    for key, value in data.items():
        fmt = formatters.get(key, str)
        rows.append([key, fmt(value)])

    table = Table(rows, colWidths=[2.6 * inch, 3.4 * inch])
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#444444")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f6f6f8")]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
    ]))
    return table


def _alpha_factor_value(value: Any) -> str:
    """Formatter for one Alpha Factors scorecard cell -- most values are
    plain numbers/strings, but "P/E vs Own History"/"P/B vs Own History"
    (see alpha_factors.py) are small nested dicts (current multiple,
    historical average, %, signal), which need their own compact
    rendering rather than falling through to Python's raw dict repr."""
    if value is None:
        return "Unavailable"
    if isinstance(value, dict):
        pct = value.get("vs_history_pct")
        signal = value.get("signal")
        if pct is not None and signal:
            return (
                f"{value.get('current')}x vs {value.get('historical_avg')}x "
                f"{value.get('years_used')}yr avg ({pct:+.1f}%, {signal})"
            )
        return str(value)
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def _sensitivity_table(df, symbol: str = "$") -> Optional[Table]:
    if df is None or df.empty:
        return None

    header = ["WACC \\ Growth"] + [f"{c:.0%}" if isinstance(c, float) else str(c) for c in df.columns]
    rows = [header]
    for idx, row in df.iterrows():
        label = f"{idx:.1%}" if isinstance(idx, float) else str(idx)
        rows.append([label] + [f"{symbol}{v:,.2f}" for v in row])

    table = Table(rows)
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#eeeeee")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def build_pdf_report(report_data: dict) -> bytes:
    """Renders report_data (from report_data_builder.py, with a
    "narrative" key merged in from narrative_builder.py) into PDF
    bytes suitable for st.download_button."""

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=LETTER,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )

    overview = report_data["company_overview"]
    narrative = report_data.get("narrative", {})
    recommendation = report_data["recommendation"]
    symbol = report_data.get("currency_symbol", "$")

    story = []

    # ---------------- Title ----------------
    story.append(Paragraph(
        f"{overview['name']} ({report_data['ticker']})", _TITLE_STYLE
    ))
    story.append(Paragraph(
        f"Equity Research Report &mdash; Generated by FinSight AI on "
        f"{datetime.now(timezone.utc).strftime('%B %d, %Y')}",
        _SUBTITLE_STYLE,
    ))

    # ---------------- Risk disclaimer ----------------
    # Placed near the top, next to the recommendation banner it
    # qualifies -- not buried in a footer where it'd be easy to skip.
    story.append(Paragraph(
        "<b>This report is a predictive analytical signal generated by an automated "
        "research tool, not personalized investment advice.</b> Past performance and "
        "model outputs are not guarantees of future results. Consult a licensed "
        "financial advisor before making investment decisions.",
        _DISCLAIMER_STYLE,
    ))

    # ---------------- Recommendation banner ----------------
    rating = recommendation["rating"]
    rating_color = _RATING_COLORS.get(rating, colors.black)
    story.append(Paragraph(
        f'Recommendation: <font color="{rating_color.hexval()}"><b>{rating}</b></font>',
        _RECOMMENDATION_STYLE,
    ))
    story.append(Paragraph(recommendation["basis"], _BODY_STYLE))
    if recommendation.get("composite_score") is not None:
        relative_score = recommendation.get("relative_score")
        story.append(_dict_table({
            "DCF Score": f"{recommendation['dcf_score']:+.1f}",
            "Relative Valuation Score": f"{relative_score:+.1f}" if relative_score is not None else "N/A",
            "Composite Score": f"{recommendation['composite_score']:+.1f}",
        }))
    if recommendation.get("confidence_flag"):
        story.append(Paragraph(
            f'<font color="#9a6700"><b>Low-confidence signal:</b></font> {recommendation["confidence_flag"]}',
            _BODY_STYLE,
        ))

    # ---------------- 1. Executive Summary ----------------
    story.append(Paragraph("1. Executive Summary", _SECTION_STYLE))
    story.append(Paragraph(narrative.get("Executive Summary", "Not available."), _BODY_STYLE))

    # ---------------- 2. Company Overview ----------------
    story.append(Paragraph("2. Company Overview", _SECTION_STYLE))
    overview_facts = {
        "Sector": overview["sector"],
        "Industry": overview["industry"],
        "Market Cap": _fmt_number(overview["market_cap"], prefix=symbol),
        "Employees": f"{overview['employees']:,}" if isinstance(overview["employees"], (int, float)) else "Unavailable",
        "Website": overview["website"] or "Unavailable",
    }
    story.append(_dict_table(overview_facts))
    story.append(Spacer(1, 8))
    story.append(Paragraph(overview["business_summary"] or "No business description available.", _BODY_STYLE))

    # ---------------- 3. Business Analysis ----------------
    story.append(Paragraph("3. Business Analysis", _SECTION_STYLE))
    story.append(Paragraph(narrative.get("Business Analysis", "Not available."), _BODY_STYLE))

    # ---------------- 4. Financial Statement Analysis ----------------
    story.append(Paragraph("4. Financial Statement Analysis", _SECTION_STYLE))
    fsa = report_data["financial_statement_analysis"]
    story.append(_dict_table(fsa, {k: (lambda v: _fmt_number(v, prefix=symbol)) for k in fsa}))

    # ---------------- 5. Ratio Analysis ----------------
    story.append(Paragraph("5. Ratio Analysis", _SECTION_STYLE))
    ratios = report_data["ratio_analysis"]
    ratio_formatters = {
        "Operating Margin (%)": _fmt_percent,
        "Net Margin (%)": _fmt_percent,
        "Return on Equity (%)": _fmt_percent,
    }
    story.append(_dict_table(ratios, ratio_formatters))

    # ---------------- 6. Growth Analysis ----------------
    story.append(Paragraph("6. Growth Analysis", _SECTION_STYLE))
    growth = {k: v for k, v in report_data["growth_analysis"].items()}
    growth_formatters = {k: _fmt_percent for k in growth if "%" in k}
    story.append(_dict_table(growth, growth_formatters))

    # ---------------- 7. Valuation Analysis (DCF) ----------------
    story.append(Paragraph("7. Valuation Analysis (DCF)", _SECTION_STYLE))
    valuation = report_data["valuation_analysis"]
    sensitivity_df = valuation.get("sensitivity_table")
    relative_valuation = valuation.get("relative_valuation")
    monte_carlo = valuation.get("monte_carlo")
    ml_classifier = valuation.get("ml_classifier")
    wacc_floor_note = valuation.get("WACC Floor Note")

    if valuation.get("DCF Available") is False:
        story.append(Paragraph(
            f'<font color="#cf222e"><b>DCF Not Applicable.</b></font> '
            f'{valuation.get("DCF Unavailable Reason") or "FCFF-DCF could not produce a meaningful intrinsic value for this company."}',
            _BODY_STYLE,
        ))
    else:
        valuation_fields = {
            k: v for k, v in valuation.items()
            if k not in ("sensitivity_table", "relative_valuation", "WACC Floor Note",
                         "DCF Available", "DCF Unavailable Reason", "monte_carlo", "ml_classifier")
        }
        valuation_formatters = {
            "Enterprise Value": lambda v: _fmt_number(v, prefix=symbol),
            "Equity Value": lambda v: _fmt_number(v, prefix=symbol),
            "Intrinsic Value (per share)": lambda v: _fmt_number(v, prefix=symbol),
            "Current Price": lambda v: _fmt_number(v, prefix=symbol),
            "Upside (%)": _fmt_percent,
            "WACC": lambda v: _fmt_percent(v * 100) if isinstance(v, (int, float)) else v,
            "Raw WACC": lambda v: _fmt_percent(v * 100) if isinstance(v, (int, float)) else v,
            "Terminal Growth Rate": lambda v: _fmt_percent(v * 100) if isinstance(v, (int, float)) else v,
        }
        story.append(_dict_table(valuation_fields, valuation_formatters))
        if wacc_floor_note:
            story.append(Spacer(1, 6))
            story.append(Paragraph(
                f'<font color="#9a6700"><b>Note:</b></font> {wacc_floor_note}',
                _BODY_STYLE,
            ))

        sensitivity_table = _sensitivity_table(sensitivity_df, symbol)
        if sensitivity_table is not None:
            story.append(Spacer(1, 10))
            story.append(Paragraph("Intrinsic Value Sensitivity (WACC x Terminal Growth)", _BODY_STYLE))
            story.append(sensitivity_table)

        if monte_carlo:
            story.append(Spacer(1, 10))
            story.append(Paragraph(
                f"Monte Carlo Simulation ({monte_carlo['n_samples']:,} samples over growth/WACC/terminal "
                f"growth) -- a distribution, not a single point estimate:",
                _BODY_STYLE,
            ))
            mc_fields = {
                "Mean Intrinsic Value": f"{symbol}{monte_carlo['mean']:,.2f}",
                "Median Intrinsic Value": f"{symbol}{monte_carlo['median']:,.2f}",
                "Std Dev": f"{symbol}{monte_carlo['std_dev']:,.2f}",
                "25th-75th Percentile": f"{symbol}{monte_carlo['p25']:,.2f} - {symbol}{monte_carlo['p75']:,.2f}",
                "90% Confidence Interval": f"{symbol}{monte_carlo['ci_lower']:,.2f} - {symbol}{monte_carlo['ci_upper']:,.2f}",
                "Probability Undervalued": f"{monte_carlo['prob_undervalued']:.1%}",
            }
            story.append(_dict_table(mc_fields))

        if ml_classifier:
            story.append(Spacer(1, 10))
            story.append(Paragraph(
                f"ML Valuation Classifier (informational only -- not part of the recommendation "
                f"above; no accuracy track record yet):",
                _BODY_STYLE,
            ))
            top_prob = ml_classifier["probabilities"].get(ml_classifier["verdict"], 0)
            ml_fields = {
                "Verdict": ml_classifier["verdict"],
                "Confidence": f"{top_prob:.1%}",
                "Model": ml_classifier["model_name"].replace("_", " ").title(),
            }
            story.append(_dict_table(ml_fields))

    if relative_valuation:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Relative Valuation Cross-Check", _BODY_STYLE))
        rv_fields = {
            "Metric": relative_valuation["metric"],
            "Current Multiple": f"{relative_valuation['current_ev_ebitda']:.2f}x",
            f"{relative_valuation['years_used']}yr Historical Average": f"{relative_valuation['historical_avg_ev_ebitda']:.2f}x",
            "vs. Own History": f"{relative_valuation['vs_history_pct']:+.1f}%",
            "Signal": relative_valuation["signal"].title(),
            "Method": relative_valuation["method"],
        }
        story.append(_dict_table(rv_fields))

    # ---------------- 8. Alpha Factors Scorecard ----------------
    # Display-only, same non-negotiable boundary as the ML classifier
    # and relative-valuation cross-check above -- see alpha_factors.py's
    # own module docstring. Each category's dict may carry an
    # underscore-prefixed "_..._note"/"_..._definition" key (e.g. the
    # .NS-ticker gating explanation) -- rendered as a small caption
    # below that category's table, not as a numeric row.
    alpha_factors = valuation.get("alpha_factors")
    if alpha_factors:
        story.append(Paragraph("8. Alpha Factors Scorecard", _SECTION_STYLE))
        story.append(Paragraph(
            "A transparent set of financial, valuation, quality, market, risk, "
            "sentiment, and macro signals shown for context alongside the DCF above "
            "&mdash; informational only, not part of the Buy/Hold/Sell recommendation.",
            _BODY_STYLE,
        ))
        category_labels = {
            "financial": "Financial", "quality": "Quality", "valuation": "Valuation",
            "market": "Market", "risk": "Risk", "sentiment": "Sentiment", "macro": "Macro",
        }
        for key, label in category_labels.items():
            category = alpha_factors.get(key) or {}
            fields = {k: v for k, v in category.items() if not k.startswith("_")}
            note = next(
                (v for k, v in category.items() if k.startswith("_") and isinstance(v, str)), None,
            )
            if not fields:
                continue
            story.append(Spacer(1, 8))
            story.append(Paragraph(label, _BODY_STYLE))
            story.append(_dict_table(fields, {k: _alpha_factor_value for k in fields}))
            if note:
                story.append(Spacer(1, 4))
                story.append(Paragraph(f'<font size="8" color="#888888">{note}</font>', _BODY_STYLE))

    # ---------------- 9. Market and Earnings Analysis ----------------
    story.append(Paragraph("9. Market and Earnings Analysis", _SECTION_STYLE))
    market = report_data["market_earnings_snapshot"]
    market_fields = {
        "Current Price": _fmt_number(market["current_price"], prefix=symbol),
        "Market Cap": _fmt_number(market["market_cap"], prefix=symbol),
        "Management Sentiment (SEC filings)": f"{market['sentiment_label']} ({market['sentiment_confidence']})",
        "Market/Media Sentiment (recent news)": f"{market['news_sentiment_label']} ({market['news_sentiment_confidence']})",
    }
    story.append(_dict_table(market_fields))
    if report_data.get("filing_evidence_note"):
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            f'<font color="#9a6700"><b>Note:</b></font> {report_data["filing_evidence_note"]}',
            _BODY_STYLE,
        ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(narrative.get("Market and Earnings Analysis", "Not available."), _BODY_STYLE))

    # ---------------- 10. Risk Analysis ----------------
    story.append(Paragraph("10. Risk Analysis", _SECTION_STYLE))
    story.append(Paragraph(narrative.get("Risk Analysis", "Not available."), _BODY_STYLE))

    # ---------------- News Sources Used in This Analysis ----------------
    # Deliberately separate from References (section 13): shows EVERY
    # article retrieved, not just the ones the model chose to cite, so
    # a reader can judge whether the news selection looks reasonable
    # (e.g. "did it miss an obvious recent story?") independent of the
    # LLM's own choices. Placed near Risk/Market Analysis since that's
    # what this news fed into, not buried at the end with References.
    news_sources = report_data.get("news_sources", {})
    if news_sources.get("total_retrieved"):
        story.append(Paragraph("News Sources Used in This Analysis", _SECTION_STYLE))
        story.append(Paragraph(
            f"{news_sources['total_retrieved']} articles retrieved (last 60 days), "
            f"{news_sources['total_selected']} selected for analysis above.",
            _BODY_STYLE,
        ))
        story.append(Spacer(1, 6))
        for article in news_sources["all_articles"]:
            marker = "[Used]" if article["used_in_analysis"] else "[Retrieved, not used]"
            text = (
                f'<font size="8">{marker}</font> <b>{article["headline"]}</b><br/>'
                f'{article["source"]} &mdash; {article["date"]} &mdash; '
                f'<a href="{article["url"]}">{article["url"]}</a>'
            )
            story.append(Paragraph(text, _BODY_STYLE))
    else:
        # Always say something here, even with zero coverage -- silence
        # would look like the section was forgotten, not that there's
        # genuinely no news.
        story.append(Paragraph("News Sources Used in This Analysis", _SECTION_STYLE))
        story.append(Paragraph(
            "No recent news coverage was found for this company.",
            _BODY_STYLE,
        ))

    # ---------------- 11. Investment Thesis ----------------
    story.append(Paragraph("11. Investment Thesis", _SECTION_STYLE))
    story.append(Paragraph(narrative.get("Investment Thesis", "Not available."), _BODY_STYLE))

    # ---------------- 12. Recommendation (recap) ----------------
    story.append(Paragraph("12. Buy / Hold / Sell Recommendation", _SECTION_STYLE))
    story.append(Paragraph(
        f'<font color="{rating_color.hexval()}"><b>{rating}</b></font> &mdash; {recommendation["basis"]}',
        _BODY_STYLE,
    ))
    if recommendation.get("confidence_flag"):
        story.append(Paragraph(
            f'<font color="#9a6700"><b>Low-confidence signal:</b></font> {recommendation["confidence_flag"]}',
            _BODY_STYLE,
        ))

    # ---------------- 13. Confidence Scores ----------------
    story.append(Paragraph("13. Confidence Scores", _SECTION_STYLE))
    confidence = report_data["confidence_scores"]
    confidence_formatters = {k: _fmt_percent for k in confidence if "%" in k}
    story.append(_dict_table(confidence, confidence_formatters))

    # ---------------- 14. References ----------------
    story.append(Paragraph("14. References", _SECTION_STYLE))
    for i, ref in enumerate(report_data["references"], 1):
        text = f"[{i}] {ref['type']}: {ref['label']}"
        if ref.get("url"):
            text += f' &mdash; <a href="{ref["url"]}">{ref["url"]}</a>'
        story.append(Paragraph(text, _BODY_STYLE))

    # ---------------- 15. Institutional Consensus Score ----------------
    story.append(Paragraph("15. Institutional Consensus Score", _SECTION_STYLE))
    consensus = (report_data.get("institutional_consensus") or {}).get("recommendation_consensus")
    if not consensus:
        story.append(Paragraph(
            "No institutional analyst coverage was found for this ticker, so no "
            "consensus comparison is available. This does not affect FinSight's "
            "own recommendation above.",
            _BODY_STYLE,
        ))
    else:
        story.append(Paragraph(
            f'<font size="16"><b>{consensus["score"]}%</b></font> &mdash; '
            f'<b>{consensus["label"]}</b>',
            _BODY_STYLE,
        ))
        story.append(Paragraph(consensus["methodology"], _BODY_STYLE))
        if consensus.get("low_sample_size") and consensus.get("sample_size_note"):
            story.append(Paragraph(consensus["sample_size_note"], _DISCLAIMER_STYLE))
        story.append(Spacer(1, 8))

        story.append(Paragraph("Market Consensus", _BODY_STYLE))
        consensus_rows = [["Firm", "Date", "Raw Grade", "Rating"]] + [
            [r["firm"], r.get("date") or "--", r.get("raw_grade") or "--", r["rating"]]
            for r in consensus["institutional_ratings"]
        ]
        consensus_table = Table(consensus_rows, colWidths=[2.6 * inch, 1.1 * inch, 1.8 * inch, 1.2 * inch])
        row_styles = [
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
        for i, r in enumerate(consensus["institutional_ratings"], start=1):
            color = _RATING_COLORS.get(r["rating"].title(), colors.black)
            row_styles.append(("TEXTCOLOR", (3, i), (3, i), color))
            row_styles.append(("FONTNAME", (3, i), (3, i), "Helvetica-Bold"))
        consensus_table.setStyle(TableStyle(row_styles))
        story.append(consensus_table)

        story.append(Spacer(1, 8))
        story.append(Paragraph(
            f'FinSight Recommendation: <font color="{rating_color.hexval()}">'
            f'<b>{consensus["finsight_rating"]}</b></font>',
            _BODY_STYLE,
        ))
        story.append(Paragraph(f"Market Summary: {consensus['summary']}", _BODY_STYLE))

    doc.build(story)
    return buffer.getvalue()
