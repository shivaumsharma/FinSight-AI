"""
model_consensus.py

"Model Compare": on demand, for an already-completed report, ask 3
independent models to weigh in on the SAME already-computed evidence
(recommendation, valuation, sentiment) and give their own Buy/Hold/Sell
call, then tally a consensus. This does NOT re-run SEC/RAG/DCF three
times -- the underlying evidence doesn't change, only which model is
interpreting it does, so this stays cheap and fast (three short LLM
calls in parallel) rather than three full report pipelines.

All 3 models are genuinely different underlying lineages (Meta, OpenAI
open-weight, Alibaba/Qwen), all already reachable through the existing
Groq-backed HostedProvider with zero new signup or API key -- confirmed
against the live account's real /v1/models list, not assumed from
training data (Groq's catalog changes; e.g. Mixtral is gone).
"""

import re
from concurrent.futures import ThreadPoolExecutor
from typing import List

from app.core.llm_provider import HostedProvider, LLMProviderError
from app.core.research_context import ResearchContext
from app.reporting.report_data_builder import derive_recommendation
from app.tools.valuation_tool import ValuationTool

CONSENSUS_MODELS = [
    {"label": "Llama 3.3 70B", "model": "llama-3.3-70b-versatile"},
    {"label": "GPT-OSS 120B", "model": "openai/gpt-oss-120b"},
    {"label": "Qwen3.6 27B", "model": "qwen/qwen3.6-27b"},
]

_VALID_RATINGS = {"buy", "hold", "sell"}

_RATING_RE = re.compile(r"RATING:\s*(\w+)", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*(\d+)", re.IGNORECASE)
_REASONING_RE = re.compile(r"REASONING:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _fmt(value) -> str:
    return str(value) if value not in (None, "") else "Unavailable"


def _build_prompt(report_data: dict, ticker: str) -> str:
    """Mirrors narrative_builder.py's _build_prompt data-block style
    (same field paths, same "DATA is your only source" framing) but at
    a fraction of the length -- this only needs enough evidence for an
    independent rating, not five narrative sections."""
    recommendation = report_data.get("recommendation", {})
    valuation = report_data.get("valuation_analysis", {})
    market = report_data.get("market_earnings_snapshot", {})

    data_block = f"""TICKER: {ticker}
VALUATION: Intrinsic value {_fmt(valuation.get('Intrinsic Value (per share)'))}, current price {_fmt(valuation.get('Current Price'))}, upside {_fmt(valuation.get('Upside (%)'))}%
MANAGEMENT SENTIMENT (SEC filing tone): {_fmt(market.get('sentiment_label'))} ({_fmt(market.get('sentiment_confidence'))})
MARKET/MEDIA SENTIMENT (recent news tone): {_fmt(market.get('news_sentiment_label'))} ({_fmt(market.get('news_sentiment_confidence'))})
EXISTING DCF-BASED RECOMMENDATION: {_fmt(recommendation.get('rating'))} -- {_fmt(recommendation.get('basis'))}"""

    return f"""You are an equity analyst giving a second opinion, independent of the recommendation already shown to you.

DATA (your only source of information):
{data_block}

Give YOUR OWN independent Buy/Hold/Sell call based on this data -- agreeing with the existing recommendation is fine if the data supports it, but do not just restate it without judgment.

Respond in EXACTLY this format, nothing else:
RATING: <Buy, Hold, or Sell>
CONFIDENCE: <0-100>
REASONING: <one sentence, tied to a specific number above>"""


def _parse_opinion(raw_text: str) -> dict:
    """Never raises -- a response that doesn't match the requested
    format degrades to the same "Insufficient Data" rating the rest of
    this app already uses for a low-confidence/unusable signal (see
    RecommendationData's rating type), not a new error state the
    frontend would need to special-case."""
    rating_match = _RATING_RE.search(raw_text)
    rating_word = rating_match.group(1).lower() if rating_match else ""
    if rating_word not in _VALID_RATINGS:
        return {"rating": "Insufficient Data", "confidence": None, "reasoning": "Model response could not be parsed."}

    confidence_match = _CONFIDENCE_RE.search(raw_text)
    confidence = None
    if confidence_match:
        parsed = int(confidence_match.group(1))
        confidence = max(0, min(100, parsed))

    reasoning_match = _REASONING_RE.search(raw_text)
    reasoning = reasoning_match.group(1).strip().splitlines()[0][:300] if reasoning_match else ""

    return {"rating": rating_word.capitalize(), "confidence": confidence, "reasoning": reasoning or "No reasoning provided."}


def _get_one_opinion(entry: dict, prompt: str) -> dict:
    label, model = entry["label"], entry["model"]
    try:
        provider = HostedProvider(model=model)
        raw = provider.generate(prompt, max_new_tokens=150)
        parsed = _parse_opinion(raw)
    except LLMProviderError:
        # Same per-item isolation used everywhere else in this app
        # (watchlist quotes, corporate actions, indices) -- one
        # model's outage must not blank out the other two opinions.
        parsed = {"rating": "Insufficient Data", "confidence": None, "reasoning": "This model was unreachable."}

    return {"label": label, "model": model, **parsed}


def get_model_opinions(report_data: dict, ticker: str) -> List[dict]:
    """Fires all 3 models concurrently (I/O-bound HTTP calls, not CPU
    work) so total latency stays close to one call's latency rather
    than the sum of three -- each with its own HostedProvider instance
    (not the get_llm_provider() singleton), since that singleton's
    usage-accumulation isn't safe for concurrent calls and these 3
    calls need 3 different `model` values anyway."""
    prompt = _build_prompt(report_data, ticker)
    with ThreadPoolExecutor(max_workers=len(CONSENSUS_MODELS)) as pool:
        return list(pool.map(lambda entry: _get_one_opinion(entry, prompt), CONSENSUS_MODELS))


def compute_consensus(opinions: List[dict]) -> dict:
    """Majority rating among opinions that actually produced a real
    Buy/Hold/Sell call -- an "Insufficient Data" opinion (parse
    failure or unreachable model) doesn't count toward or against any
    rating, same as this app treats a missing signal elsewhere rather
    than defaulting it to Hold."""
    counts = {"Buy": 0, "Hold": 0, "Sell": 0}
    for opinion in opinions:
        if opinion["rating"] in counts:
            counts[opinion["rating"]] += 1

    total_voting = sum(counts.values())
    if total_voting == 0:
        return {"rating": "Insufficient Data", "agree_count": 0, "total": len(opinions)}

    top_rating = max(counts, key=lambda r: counts[r])
    return {"rating": top_rating, "agree_count": counts[top_rating], "total": len(opinions)}


def get_stock_model_opinions(ticker: str) -> dict:
    """Model Compare for the stock-detail-page: same 3-model second
    opinion as the job-based version above, but there's no completed
    research job to read report_data from here -- this page never runs
    one. Uses the same lightweight, LLM/RAG-free ValuationTool-only
    path as build_stock_insights (stock_score.py) to get a
    recommendation, then shapes it into the report_data keys
    _build_prompt already expects. SEC/news sentiment stay "Unavailable"
    (via _fmt) since no sentiment tool runs in this path -- the 3
    models just have one less input than the job-based version, not
    broken input. Raises TickerNotFoundError (via ValuationTool's
    internal MarketDataTool run) for a bad ticker, same as every other
    ticker-keyed endpoint in main.py."""
    context = ResearchContext(ticker=ticker, question=f"Should I invest in {ticker}?")
    ValuationTool().run(context)

    valuation_results = context.valuation_results or {}
    currency = (context.company_info or {}).get("currency") or "USD"
    recommendation = derive_recommendation(valuation_results, None, None, currency=currency)

    report_data = {
        "recommendation": recommendation,
        "valuation_analysis": {
            "Intrinsic Value (per share)": valuation_results.get("intrinsic_value"),
            "Current Price": valuation_results.get("current_price"),
            "Upside (%)": valuation_results.get("upside_percent"),
        },
        "market_earnings_snapshot": {},
    }

    opinions = get_model_opinions(report_data, ticker)
    return {"models": opinions, "consensus": compute_consensus(opinions)}
