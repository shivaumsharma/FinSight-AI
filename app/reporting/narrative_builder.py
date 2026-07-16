"""
narrative_builder.py

Writes the five narrative sections of the report (Executive Summary,
Business Analysis, Market and Earnings Analysis, Risk Analysis,
Investment Thesis) in a single combined LLM call, then splits the
output by heading into a {section_name: text} dict. Every other
report section is built deterministically from real computed data --
see report_data_builder.py.

Deliberately one call, not five: on the current CPU-bound local
model, five separate generation calls would multiply an already slow
per-report latency several times over. The tradeoff is asking more of
a small model in one pass -- the same drift-handling already needed
for the old 5-section report (meta-commentary, degenerate repeated
output once the model runs past what it actually has to say) is
reused and generalized here for the new section set.

Two failure modes found by inspecting real output (not hypothetical):
1. The prompt asked the model to stay "consistent with the DETERMINED
   RECOMMENDATION" but gave it no concrete guidance on HOW -- a Sell
   rating still produced "the overall outlook remains positive" and
   "represents a promising investment opportunity" verbatim. Fixed
   two ways: rating-specific framing guidance in the prompt (below),
   and a deterministic post-generation guardrail that appends a
   corrective note to any section using known contradiction-prone
   language -- prompting alone isn't reliable enough on its own with
   this model to skip the second layer.
2. Citation coverage regressed after this file was rewritten: the
   old prompt_builder.py instructed the model to tag sentences with
   "[Evidence N]" when based on specific retrieved evidence --
   citation_evaluator.py's primary detection method depends on that
   tag, and the rewrite dropped the instruction. Restored below.
"""

import re
from typing import Dict

from app.core.research_context import ResearchContext
from app.core.llm_provider import get_shared_generator
from app.reporting.news_client import RISK_CATEGORIES

NARRATIVE_SECTIONS = [
    "Executive Summary",
    "Business Analysis",
    "Market and Earnings Analysis",
    "Risk Analysis",
    "Investment Thesis",
]

_DRIFT_MARKERS = (
    "please provide", "assessment:", "response:", "suggestions:",
    "let me know", "your response should", "for example:",
    "thank you", "feedback on", "your insights will help",
    # Generic disclaimer/meta-commentary observed trailing the last
    # section in real output (e.g. "Please note that the determination
    # of whether to buy, hold, or sell involves subjective judgment...
    # Always conduct thorough research before making investment
    # decisions.") -- not part of the report, and the original marker
    # list didn't catch this specific phrasing.
    "please note that", "always conduct thorough research",
    "individual circumstances", "before making investment decisions",
)

# Phrases observed in real output that flatly contradict a Sell/Buy
# rating despite the prompt instructing consistency. Not exhaustive --
# a heuristic guardrail, not a semantic contradiction detector -- but
# it catches the specific failure mode already seen in practice and
# costs nothing to check.
_BULLISH_PHRASES = (
    "promising investment", "compelling combination", "compelling opportunity",
    "poised for sustained expansion", "substantial potential for long-term appreciation",
    "attractive opportunity", "outlook remains positive", "strong buy",
    "significant upside potential", "poised for growth",
)

_BEARISH_PHRASES = (
    "poor investment", "avoid this stock", "substantial risk of decline",
    "warrants caution against investment", "significant downside risk",
    "unattractive opportunity", "outlook remains negative",
)


def _looks_like_drift(line: str) -> bool:
    stripped = line.strip()
    # A standalone "---"/"___" horizontal rule is a strong signal the
    # model has finished the actual report and is about to add
    # meta-commentary after it -- real report prose doesn't produce
    # a bare separator line like this.
    if stripped and all(ch in "-_=" for ch in stripped) and len(stripped) >= 3:
        return True
    lowered = stripped.lower()
    if any(marker in lowered for marker in _DRIFT_MARKERS):
        return True
    # Degenerate emoji/symbol runs (each character here is well past
    # the ASCII/basic-punctuation range).
    non_ascii = sum(1 for ch in line if ord(ch) > 0x2500)
    if non_ascii > 5:
        return True
    return False


def _framing_guidance(rating: str) -> str:
    if rating == "Sell":
        return (
            "FinSight's own rating is Sell. Frame every section in that light: "
            'do NOT call this a "promising investment," "compelling opportunity," '
            'or describe the outlook as "positive" -- lead with the valuation gap '
            "and risks, and treat any growth/profitability strengths as context, "
            "not a reason to be bullish."
        )
    if rating == "Buy":
        return (
            "FinSight's own rating is Buy. Frame every section in that light -- "
            "still note real risks from the DATA, but the overall framing should "
            "support the Buy case, not undercut it with unexplained caution."
        )
    if rating == "Hold":
        return (
            "FinSight's own rating is Hold. Frame every section as balanced -- "
            "avoid strong bullish or bearish language in either direction."
        )
    return "No valuation is available, so do not assert a bullish or bearish stance."


def _news_data_block(context: ResearchContext) -> str:
    """
    Numbered [News N] list (1-based within context.news_selected,
    same convention as [Evidence N] within context.citations) grouped
    by risk category, explicitly stating when a category has no
    matching news -- so the model is told to say "no coverage found"
    rather than invent generic risk language to fill the gap.
    """
    selected = context.news_selected or []

    if not selected:
        return (
            "NEWS: No recent news coverage was found for this company. "
            "Do not invent litigation, regulatory, competitive, or macro "
            "risk details -- state plainly that no recent news-based risk "
            "signals were available."
        )

    lines = [f"NEWS ({len(context.news_articles)} articles retrieved, {len(selected)} selected below):"]
    for i, article in enumerate(selected, 1):
        lines.append(f"[News {i}] ({article['date']}, {article['source']}): {article['headline']} -- {article['summary'][:200]}")

    lines.append("")
    lines.append("News by risk category (cite the specific [News N] item(s) for any risk you mention; state 'no recent news found' for a category with none listed):")
    for category in RISK_CATEGORIES:
        matches = [i for i, a in enumerate(selected, 1) if category in a.get("categories", [])]
        if matches:
            lines.append(f"- {category.title()}: " + ", ".join(f"[News {i}]" for i in matches))
        else:
            lines.append(f"- {category.title()}: none found")

    return "\n".join(lines)


def _build_prompt(context: ResearchContext, report_data: dict) -> str:

    overview = report_data["company_overview"]
    growth = report_data["growth_analysis"]
    valuation = report_data["valuation_analysis"]
    market = report_data["market_earnings_snapshot"]
    recommendation = report_data["recommendation"]

    data_block = f"""COMPANY: {overview['name']} ({report_data['ticker']}) -- {overview['sector']} / {overview['industry']}
BUSINESS: {overview['business_summary'] or 'Not available.'}

GROWTH: Revenue growth {growth['Revenue Growth (%)']}%, trend: {growth['Revenue Trend']}
VALUATION: Intrinsic value {valuation['Intrinsic Value (per share)']}, current price {valuation['Current Price']}, upside {valuation['Upside (%)']}%
MARKET: Current price {market['current_price']}
MANAGEMENT SENTIMENT (from SEC filing tone): {market['sentiment_label']} ({market['sentiment_confidence']})
MARKET/MEDIA SENTIMENT (from recent news tone): {market['news_sentiment_label']} ({market['news_sentiment_confidence']})
DETERMINED RECOMMENDATION: {recommendation['rating']} -- {recommendation['basis']}
{"CONFIDENCE CAVEAT: " + recommendation["confidence_flag"] if recommendation.get("confidence_flag") else ""}

{_news_data_block(context)}

RESEARCH CONTEXT (financials, numbered evidence, citations):
{context.research_summary}"""

    section_template = "\n\n".join(f"# {s}\n[...]" for s in NARRATIVE_SECTIONS)

    return f"""You are a financial analyst writing sections of an institutional-style equity research report.

DATA below is your only source of information -- treat it as real and sufficient. If a specific figure says "Unavailable", skip that detail rather than refusing to write.

{data_block}

{_framing_guidance(recommendation['rating'])}

Fill in the template below, replacing each [...] with 2-4 sentences based only on the DATA above. Keep every "# " heading exactly as shown, in this order:

{section_template}

Rules:
- Every sentence must reference a specific number, fact, or numbered evidence/news item from the DATA above -- no generic filler like "continues to demonstrate steady progress" or "well-positioned for future growth" that isn't tied to something concrete in the DATA.
- When a sentence is based on a specific item from RESEARCH CONTEXT's evidence, tag the end of it with that item's number, e.g. "...enterprise adoption accelerated [Evidence 4]." When based on a specific news item, tag it the same way, e.g. "...facing a antitrust probe [News 2]."
- Risk Analysis specifically: use the "News by risk category" list above. For each category with news, cite the specific [News N] item(s) for that risk -- do not write generic risk boilerplate ("faces competition," "regulatory uncertainty") without a citation. For a category listed as "none found," say plainly that no recent news-based signal was found for that risk type -- do not invent one.
- Market and Earnings Analysis specifically: if MANAGEMENT SENTIMENT and MARKET/MEDIA SENTIMENT differ, say so explicitly -- that divergence is itself worth noting, not something to paper over by picking one.
- No outside knowledge about the company's products, competitors, or history beyond what's in the DATA. Do not invent management quotes or numbers.
- The Investment Thesis section MUST match the DETERMINED RECOMMENDATION and the framing guidance above -- do not contradict either.
- Keep the whole response under 500 words.
- Write all five sections. Stop immediately after Investment Thesis -- no feedback, self-assessment, or commentary about this task afterward.
"""


def _contradiction_note(rating: str, basis: str) -> str:
    return (
        f"[Note: this section's framing has been flagged as potentially "
        f"inconsistent with FinSight's own {rating} rating. FinSight's "
        f"determined position is: {basis}]"
    )


def _apply_contradiction_guardrail(sections: Dict[str, str], recommendation: dict) -> Dict[str, str]:
    """
    Deterministic defense-in-depth, not a replacement for the prompt
    guidance above: prompting this model to stay consistent with the
    rating measurably doesn't always work (verified against real
    output -- see module docstring), so this scans the generated text
    for known contradiction-prone phrases and appends a clear,
    deterministic corrective note wherever they appear, guaranteeing
    a reader never sees an unresolved contradiction even when the
    model itself drifts.
    """
    rating = recommendation["rating"]

    if rating == "Sell":
        watch_phrases = _BULLISH_PHRASES
    elif rating == "Buy":
        watch_phrases = _BEARISH_PHRASES
    else:
        return sections

    note = _contradiction_note(rating, recommendation["basis"])

    for name, text in sections.items():
        lowered = text.lower()
        if any(phrase in lowered for phrase in watch_phrases):
            sections[name] = f"{text}\n\n{note}"

    return sections


def _split_sections(text: str) -> Dict[str, str]:
    """
    Splits `text` into {section_name: content} by heading markers
    (the heading must be alone on its own line, to avoid matching it
    as a substring of ordinary prose), trimming the model's drift the
    same way the previous single-blob report did.

    Tolerates more than the literal "# Heading" the prompt asks for:
    inspecting real output showed the model sometimes prefaces its
    response with a fake "formatting example" using **bold** headings
    before the real content, and there's no guarantee the *real*
    heading always uses "#" rather than "**" either. The original
    regex only tolerated an optional leading "#", so a real heading
    written as "**Executive Summary**" would silently fail to match
    at all -- producing exactly the "Not available for this report"
    placeholder this was meant to prevent. Now tolerant of leading/
    trailing "#", "*", ":" in any combination, while still requiring
    the line to consist of nothing else (so it can't match a heading
    name mentioned mid-sentence).
    """
    positions = []
    for section in NARRATIVE_SECTIONS:
        matches = list(re.finditer(
            rf"^[#*\s]*{re.escape(section)}[*:\s]*$", text, re.IGNORECASE | re.MULTILINE
        ))
        if matches:
            positions.append((matches[-1].start(), matches[-1].end(), section))

    positions.sort(key=lambda p: p[0])

    sections = {}
    for i, (start, end, name) in enumerate(positions):
        content_start = end
        content_end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        content = text[content_start:content_end].strip()

        kept_lines = []
        for line in content.split("\n"):
            if line.strip() and _looks_like_drift(line.strip()):
                break
            kept_lines.append(line)
        sections[name] = "\n".join(kept_lines).strip()

    return sections


def build_narrative_sections(context: ResearchContext, report_data: dict) -> Dict[str, str]:
    """
    Returns {section_name: text} for every name in NARRATIVE_SECTIONS.
    Any section the model failed to produce a heading for gets an
    explicit placeholder rather than silently vanishing from the
    report -- callers/PDF rendering can rely on every key existing.
    """
    prompt = _build_prompt(context, report_data)

    raw = get_shared_generator().generate(prompt, max_new_tokens=700)

    sections = _split_sections(raw)

    for section in NARRATIVE_SECTIONS:
        sections.setdefault(section, "Not available for this report.")

    sections = _apply_contradiction_guardrail(sections, report_data["recommendation"])

    return sections
