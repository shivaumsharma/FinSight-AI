"""
news_sentiment.py

Scores each news article individually with FinBERT, rather than
concatenating everything into one blob. A single aggregate score
would average away exactly the "mixed signals" case that matters
most: 8 positive articles and 2 negative ones should report as a
distribution the reader can see, not get diluted into one falsely-
clean net number that looks identical to genuinely uniform coverage.
"""

from typing import Dict, List, Optional

from app.nlp.finbert import FinBERT

_model = None


def _get_model() -> FinBERT:
    global _model
    if _model is None:
        _model = FinBERT()
    return _model


def score_news_sentiment(articles: List[Dict]) -> Optional[Dict]:
    """
    Returns None if there are no articles to score -- "insufficient
    news coverage" is reported upstream (report_data_builder.py), not
    faked here as a 0 or a default label. Otherwise returns per-label
    counts, a net score, an explicit "is this genuinely mixed"
    signal, and the raw per-article scores.
    """
    if not articles:
        return None

    texts = [f"{a['headline']}. {a.get('summary', '')}".strip() for a in articles]

    raw_results = _get_model().analyze_many(texts)

    per_article = []
    counts = {"positive": 0, "negative": 0, "neutral": 0}

    for article, result in zip(articles, raw_results):
        label = result["label"].lower()
        score = float(result["score"])
        counts[label] = counts.get(label, 0) + 1
        per_article.append({
            "headline": article["headline"],
            "url": article["url"],
            "label": label,
            "confidence": score,
        })

    total = len(per_article)
    dominant_label = max(counts, key=counts.get)
    dominant_share = counts[dominant_label] / total * 100

    # -100..100: positive share minus negative share, so genuinely
    # mixed coverage nets toward zero instead of being hidden behind
    # whichever label happens to hold a plurality.
    net_score = (counts["positive"] - counts["negative"]) / total * 100

    # Flagged as "mixed" only when both positive and negative each
    # have a real presence (>=25% of articles), not just one stray
    # outlier against an otherwise uniform corpus.
    is_mixed = (
        counts["positive"] > 0
        and counts["negative"] > 0
        and min(counts["positive"], counts["negative"]) / total >= 0.25
    )

    return {
        "total_articles": total,
        "counts": counts,
        "dominant_label": dominant_label,
        "dominant_share": round(dominant_share, 1),
        "net_score": round(net_score, 1),
        "is_mixed": is_mixed,
        "per_article": per_article,
    }


def build_news_sentiment_summary(result: Optional[Dict]) -> Dict:
    """Display-ready summary, matching the shape of the existing
    SentimentSummaryBuilder (Management Sentiment) so both can be
    shown side by side."""

    if not result:
        return {
            "Overall Sentiment": "Insufficient Coverage",
            "Confidence": "N/A",
            "Interpretation": "No recent news articles were found for this company.",
        }

    if result["is_mixed"]:
        counts = result["counts"]
        return {
            "Overall Sentiment": "Mixed / Conflicting",
            "Confidence": f"{counts['positive']} positive / {counts['negative']} negative / {counts['neutral']} neutral (of {result['total_articles']})",
            "Interpretation": (
                "Recent media coverage is genuinely split -- a meaningful share of "
                "articles lean positive and a meaningful share lean negative, rather "
                "than one clear tone."
            ),
        }

    label = result["dominant_label"]
    interpretation = {
        "positive": "Recent media coverage leans optimistic.",
        "negative": "Recent media coverage leans cautious or critical.",
        "neutral": "Recent media coverage is largely neutral in tone.",
    }[label]

    return {
        "Overall Sentiment": label.title(),
        "Confidence": f"{result['dominant_share']:.1f}% of {result['total_articles']} articles",
        "Interpretation": interpretation,
    }
