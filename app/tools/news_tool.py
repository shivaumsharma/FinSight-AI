"""
news_tool.py

Fetches recent company news (Finnhub), scores it with a dedicated
per-article FinBERT pass ("Market/Media Sentiment", distinct from the
existing SEC-filing-based "Management Sentiment"), and makes both the
full retrieved set and a capped, category-diverse selection available
on the context -- the former for the transparency panel, the latter
for the Risk Analysis / Market and Earnings Analysis narrative.

Always degrades gracefully: no API key, no network, or a ticker with
no recent coverage all leave context.news_articles empty rather than
raising -- report_data_builder.py / narrative_builder.py are expected
to say "insufficient news coverage" rather than invent anything.
"""

from app.core.research_context import ResearchContext
from app.reporting.news_client import fetch_company_news, select_for_analysis
from app.reporting.news_sentiment import score_news_sentiment, build_news_sentiment_summary
from .base_tool import BaseTool


class NewsTool(BaseTool):

    name = "news_tool"
    description = (
        "Fetches recent company news and scores Market/Media Sentiment separately "
        "from Management Sentiment. Feeds Risk Analysis with real, categorized news "
        "instead of generic filler. Degrades gracefully to 'no coverage' rather than "
        "erroring when no news is available."
    )

    def run(self, context: ResearchContext) -> ResearchContext:

        articles = fetch_company_news(context.ticker)

        context.news_articles = articles
        context.news_selected = select_for_analysis(
            articles,
            ticker=context.ticker,
            # news_tool is a trailing tool that always runs (see
            # agent_constants.TRAILING_TOOLS), but market_data_tool --
            # the only source of company_info -- isn't guaranteed to
            # have run first for every plan (e.g. a pure "what did
            # management say" question). select_for_analysis() already
            # degrades to ticker-only matching when this is None.
            company_name=(context.company_info or {}).get("company_name"),
        )

        context.news_sentiment = score_news_sentiment(articles)
        context.news_sentiment_summary = build_news_sentiment_summary(context.news_sentiment)

        context.record_tool(self.name)

        return context
