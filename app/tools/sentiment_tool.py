from .base_tool import BaseTool
from app.nlp.finbert import FinBERT


class SentimentTool(BaseTool):

    name = "sentiment_tool"
    description = "Runs FinBERT sentiment analysis."

    def __init__(self):
        self.model = FinBERT()

    def run(self, **kwargs):

        text = kwargs.get("text", "")

        result = self.model.predict(text)

        return result