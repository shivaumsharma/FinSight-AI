"""
query_classifier.py

Simple rule-based classifier for financial questions.
Later this can be replaced with a fine-tuned classifier.
"""

class QueryClassifier:

    def classify(self, question: str) -> str:

        q = question.lower()

        investment_keywords = [
            "invest",
            "buy",
            "sell",
            "hold",
            "recommend",
            "worth"
        ]

        valuation_keywords = [
            "valuation",
            "intrinsic",
            "dcf",
            "wacc",
            "fair value",
            "enterprise value"
        ]

        transcript_keywords = [
            "management",
            "earnings call",
            "conference call",
            "said",
            "mentioned",
            "commentary"
        ]

        financial_keywords = [
            "revenue",
            "margin",
            "eps",
            "cash flow",
            "roe",
            "income",
            "profit"
        ]

        if any(word in q for word in investment_keywords):
            return "investment"

        if any(word in q for word in valuation_keywords):
            return "valuation"

        if any(word in q for word in transcript_keywords):
            return "transcript"

        if any(word in q for word in financial_keywords):
            return "financial"

        return "general"