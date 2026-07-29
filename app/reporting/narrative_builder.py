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

import logging
import re
from datetime import date
from typing import Dict, Optional

from app.core.research_context import ResearchContext
from app.core.llm_provider import get_llm_provider
from app.core.cache import cache_get, cache_set, make_key
from app.reporting.news_client import RISK_CATEGORIES
from app.reporting.narrative_debug_log import log_narrative_call

logger = logging.getLogger(__name__)

# Content-addressed by a hash of the full prompt (see app/core/cache.py's
# module docstring for why this is safe: if any input to the prompt
# changes -- financials, valuation, sentiment, news, the DETERMINED
# RECOMMENDATION -- the prompt itself changes, so the key changes and
# it's a natural miss, never a stale hit). 24h TTL is purely a storage
# bound; it's not what keeps this correct.
NARRATIVE_CACHE_TTL_SECONDS = 24 * 3600

# How close a reported next_earnings_date has to be (in either
# direction is impossible here -- MarketDataLoader.get_next_earnings_date()
# only ever returns future/today dates -- so this is purely "how many
# days out") before it's forced into the prompt as its own prominent
# block instead of being left to compete for attention with the 249
# retrieved news headlines. Confirmed on a real MSFT run published the
# day before earnings, with the options market pricing a 7% move, that
# the narrative never mentioned the earnings date at all even though a
# news item naming it was among those selected.
EARNINGS_PROXIMITY_DAYS = 7

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


def _as_number(value) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _fmt_pct(fraction_value) -> str:
    """valuation_summary.py stores WACC/terminal growth as raw fractions
    (0.1062, not 10.62) -- see pdf_report_builder.py's own `v * 100`
    before formatting for the same reason."""
    number = _as_number(fraction_value)
    return f"{number * 100:.2f}%" if number is not None else "Unavailable"


def _growth_divergence_block(growth: dict) -> str:
    """
    FCF Growth (%) is already a bare number in the GROWTH line above --
    easy to walk past when revenue/net income are both growing right
    next to it. Confirmed on a real MSFT run: Revenue +14.93%, Net
    Income +15.54%, FCF -3.32% -- exactly the divergence a DCF (a
    free-cash-flow-based model) reacts to -- and the narrative never
    mentioned FCF at all, despite it being the actual reason the DCF
    verdict disagreed with the rest of the report. Forces it into its
    own labeled block with an explicit instruction rather than trusting
    the model to notice a negative number sitting next to two positive
    ones.
    """
    fcf_growth = _as_number(growth.get("FCF Growth (%)"))
    if fcf_growth is None or fcf_growth >= 0:
        return ""

    other_growth = (
        _as_number(growth.get("Revenue Growth (%)")),
        _as_number(growth.get("Net Income Growth (%)")),
    )
    if not any(g is not None and g > 0 for g in other_growth):
        return ""

    return (
        f"\nGROWTH DIVERGENCE: Free Cash Flow growth is {fcf_growth}% (trend: "
        f"{growth.get('FCF Trend', 'Unknown')}) while revenue and/or net income are "
        f"growing -- this divergence is very likely why a DCF (a free-cash-flow-based "
        f"model) disagrees with other valuation signals here. You MUST name this "
        f"divergence explicitly in your narrative (e.g. rising capital spending or "
        f"working capital consuming cash despite profit growth -- check NEWS/RESEARCH "
        f"CONTEXT below for the specific driver) as a likely explanation for any "
        f"valuation gap -- do not just restate the growth numbers without explaining "
        f"what the FCF decline means for the valuation."
    )


def _earnings_proximity_block(next_earnings_date) -> str:
    """
    Confirmed on a real MSFT run published the day before earnings,
    with the options market pricing a +/-7% move, that an imminent
    earnings date sitting among the 249 retrieved news headlines never
    made it into the narrative -- forced into its own labeled block
    instead, gated to EARNINGS_PROXIMITY_DAYS so a routine earnings
    date months out doesn't clutter every report.
    """
    if not next_earnings_date:
        return ""
    days_until = (next_earnings_date - date.today()).days
    if not (0 <= days_until <= EARNINGS_PROXIMITY_DAYS):
        return ""

    when = "today" if days_until == 0 else f"in {days_until} day(s)"
    return (
        f"\nUPCOMING EARNINGS: This company reports earnings on "
        f"{next_earnings_date.isoformat()} ({when}) -- imminent relative to this "
        f"report. You MUST mention this explicitly in Executive Summary and/or Market "
        f"and Earnings Analysis, and note that the valuation/recommendation above could "
        f"move materially on the results -- do not present it as if it will hold "
        f"regardless of what's reported."
    )


def _build_prompt(context: ResearchContext, report_data: dict) -> str:

    overview = report_data["company_overview"]
    growth = report_data["growth_analysis"]
    valuation = report_data["valuation_analysis"]
    market = report_data["market_earnings_snapshot"]
    recommendation = report_data["recommendation"]

    data_block = f"""COMPANY: {overview['name']} ({report_data['ticker']}) -- {overview['sector']} / {overview['industry']}
BUSINESS: {overview['business_summary'] or 'Not available.'}

GROWTH: Revenue growth {growth['Revenue Growth (%)']}%, trend: {growth['Revenue Trend']}; Free Cash Flow growth {growth['FCF Growth (%)']}%, trend: {growth.get('FCF Trend', 'Unknown')}
VALUATION: Intrinsic value {valuation['Intrinsic Value (per share)']}, current price {valuation['Current Price']}, upside {valuation['Upside (%)']}% -- DCF assumptions: WACC {_fmt_pct(valuation.get('WACC'))}, terminal growth {_fmt_pct(valuation.get('Terminal Growth Rate'))}
MARKET: Current price {market['current_price']}
MANAGEMENT SENTIMENT (from SEC filing tone): {market['sentiment_label']} ({market['sentiment_confidence']})
MARKET/MEDIA SENTIMENT (from recent news tone): {market['news_sentiment_label']} ({market['news_sentiment_confidence']})
DETERMINED RECOMMENDATION: {recommendation['rating']} -- {recommendation['basis']}
{"CONFIDENCE CAVEAT: " + recommendation["confidence_flag"] if recommendation.get("confidence_flag") else ""}
{_growth_divergence_block(growth)}
{_earnings_proximity_block(market.get('next_earnings_date'))}

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
- If a GROWTH DIVERGENCE or UPCOMING EARNINGS block appears above, you MUST address it explicitly in the section(s) it names -- these are the report's own most important facts and are not optional to skip for space.
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


_HEADING_TOKEN_PATTERN = re.compile(
    r"#{1,3}\s*\**\s*(?:" + "|".join(re.escape(s) for s in NARRATIVE_SECTIONS) + r")",
    re.IGNORECASE,
)


def _ensure_headings_start_own_line(text: str) -> str:
    """
    Confirmed via raw-output inspection (narrative_debug_log.py, on
    real GOOGL/PLTR failures): the model reliably produces the right
    headings, in the right order, with real content -- it just
    doesn't reliably insert a line break BEFORE "#", sometimes gluing
    a heading onto the end of the prior sentence, e.g. "...for
    further investment. # Business Analysis". _split_sections()'s
    regex below requires a heading to be alone on its own line, so a
    heading landing mid-line was silently left unmatched -- a parsing
    assumption that didn't hold for the model's actual (consistent)
    output shape, not a model-capability gap.

    Runs before the line-anchored split: finds every recognized
    heading token (with a literal "#" -- this doesn't touch bare
    heading-name mentions in ordinary prose) anywhere in the text and
    inserts a newline immediately before it if one isn't already
    there. Deliberately loose -- if this over-fires on a heading-like
    phrase that isn't really a heading, the existing line-anchored
    regex below still requires the resulting line to consist of
    nothing but the heading text, so a false positive here just fails
    to match downstream instead of corrupting a real section.
    """
    def _prefix_newline(match):
        start = match.start()
        if start == 0 or match.string[start - 1] == "\n":
            return match.group(0)
        return "\n" + match.group(0)

    return _HEADING_TOKEN_PATTERN.sub(_prefix_newline, text)


def _split_sections(text: str) -> Dict[str, str]:
    """
    Splits `text` into {section_name: content} by heading markers
    (the heading must be alone on its own line, to avoid matching it
    as a substring of ordinary prose), trimming the model's drift the
    same way the previous single-blob report did.

    If a heading appears more than once, the FIRST occurrence wins --
    a duplicate heading is the model drifting (restarting or repeating
    a section rather than moving on), not intentional structure, and a
    later repeat is more likely to be truncated or off-topic than the
    original. A warning is logged so a recurring duplicate can be
    noticed and investigated rather than silently resolved every time.

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

    _ensure_headings_start_own_line() runs first so a heading that
    landed mid-line in the raw output still satisfies the
    "alone on its own line" requirement below.
    """
    text = _ensure_headings_start_own_line(text)

    # Every occurrence of every heading, not just the first per name --
    # this is what lets a repeated heading act as a real boundary: the
    # first "# Executive Summary"'s content has to stop at the SECOND
    # "# Executive Summary" (or any other heading) that follows it, not
    # run past it into the duplicate's own content.
    positions = []
    for section in NARRATIVE_SECTIONS:
        for m in re.finditer(
            rf"^[#*\s]*{re.escape(section)}[*:\s]*$", text, re.IGNORECASE | re.MULTILINE
        ):
            positions.append((m.start(), m.end(), section))

    positions.sort(key=lambda p: p[0])

    sections = {}
    for i, (start, end, name) in enumerate(positions):
        content_start = end
        content_end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        content = text[content_start:content_end].strip()

        if name in sections:
            # A repeated heading is model drift, not intentional
            # structure -- the prompt asks for each section exactly
            # once. The FIRST occurrence was already kept above; a
            # later duplicate means the model restarted or repeated
            # the section, and its content is truncated or off-topic
            # more often than the original, so it's discarded rather
            # than overwriting the complete first section.
            logger.warning(
                "narrative_builder: duplicate heading %r found in model "
                "output -- keeping the first occurrence, discarding this "
                "later one as drift",
                name,
            )
            continue

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

    cache_key = make_key("narrative", context.ticker, prompt)
    raw = cache_get(cache_key)

    if raw is None:
        raw = get_llm_provider().generate(prompt, max_new_tokens=700)
        cache_set(cache_key, raw, ttl_seconds=NARRATIVE_CACHE_TTL_SECONDS)

    sections = _split_sections(raw)

    # Diagnostic-only logging (app/reporting/narrative_debug_log.py) --
    # captures the raw model output and which headings actually
    # matched BEFORE the placeholder below papers over a miss, so a
    # recurring "Not available for this report" failure can be
    # diagnosed from real output instead of guessed at. Not a fix.
    try:
        log_narrative_call(
            ticker=context.ticker,
            prompt=prompt,
            raw_output=raw,
            raw_sections=sections,
            recommendation_rating=report_data.get("recommendation", {}).get("rating"),
        )
    except Exception:
        pass

    for section in NARRATIVE_SECTIONS:
        sections.setdefault(section, "Not available for this report.")

    sections = _apply_contradiction_guardrail(sections, report_data["recommendation"])

    return sections
