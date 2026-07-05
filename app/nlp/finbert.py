from transformers import pipeline


class FinBERT:

    """
    Financial sentiment analysis using ProsusAI/finbert.
    """

    def __init__(self):

        self.model = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )

    def analyze(self, text: str):

        result = self.model(text)[0]

        return {
            "label": result["label"],
            "score": float(result["score"])
        }

    def analyze_many(self, texts):

        return self.model(texts)