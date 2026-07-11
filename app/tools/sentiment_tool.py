"""
sentiment_tool.py

Runs FinBERT sentiment analysis over the retrieved transcript
evidence (not raw kwargs["text"], which the old tool expected the
caller to supply manually and which the agent never actually
provided). If RAGTool hasn't run yet for this context, it runs it
first so sentiment always has real evidence to score.
"""

from app.core.research_context import ResearchContext
from app.nlp.finbert import FinBERT
from app.nlp.sentiment_summary import SentimentSummaryBuilder
from .base_tool import BaseTool


class SentimentTool(BaseTool):

    name = "sentiment_tool"
    description = (
        "Runs FinBERT sentiment analysis over retrieved earnings-call evidence "
        "to gauge whether management commentary is positive, negative, or "
        "neutral. Useful for questions about tone, sentiment, or management "
        "confidence."
    )

    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = FinBERT()
        return self._model

    def run(self, context: ResearchContext) -> ResearchContext:

        if not context.retrieved_chunks:
            from .rag_tool import RAGTool
            RAGTool().run(context)

        if context.retrieved_chunks:
            joined_text = "\n".join(
                chunk["text"] for chunk in context.retrieved_chunks
            )
            context.sentiment = self.model.analyze(joined_text)
            context.sentiment_summary = SentimentSummaryBuilder().build(
                context.sentiment
            )

        context.record_tool(self.name)

        return context
