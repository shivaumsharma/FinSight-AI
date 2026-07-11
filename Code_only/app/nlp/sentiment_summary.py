"""
Converts FinBERT output into a readable summary.
"""

class SentimentSummaryBuilder:

    def build(self, sentiment):

        if not sentiment:
            return {}

        label = sentiment.get("label", "Unknown")
        score = sentiment.get("score", 0)

        if label.lower() == "positive":
            interpretation = (
                "Management commentary is generally optimistic."
            )

        elif label.lower() == "negative":
            interpretation = (
                "Management commentary is generally cautious."
            )

        else:
            interpretation = (
                "Management commentary appears neutral."
            )

        return {

            "Overall Sentiment": label.title(),

            "Confidence": f"{score*100:.2f}%",

            "Interpretation": interpretation

        }