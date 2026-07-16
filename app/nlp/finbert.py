from transformers import pipeline

# Process-wide singleton, mirroring app/core/llm_provider.py. FinBERT
# is otherwise reconstructed (and its HF pipeline reloaded from disk)
# on every single request, since ResearchAgent/ToolRegistry/SentimentTool
# are all rebuilt fresh per Streamlit click.
_shared_pipeline = None


def _get_shared_pipeline():
    global _shared_pipeline
    if _shared_pipeline is None:
        _shared_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )
    return _shared_pipeline


class FinBERT:

    """
    Financial sentiment analysis using ProsusAI/finbert.
    """

    def __init__(self):

        self.model = _get_shared_pipeline()

    def analyze(self, text: str):

        # truncation=True: joined evidence from real SEC filings
        # routinely exceeds BERT's 512-token limit (unlike the old
        # hand-authored demo transcripts, which never did) -- without
        # this, transformers raises instead of truncating.
        result = self.model(text, truncation=True)[0]

        return {
            "label": result["label"],
            "score": float(result["score"])
        }

    def analyze_many(self, texts):

        return self.model(texts, truncation=True)