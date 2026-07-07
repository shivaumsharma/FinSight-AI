"""
Builds an optimized retrieval query
from the detected user intent.
"""


class RetrievalPlanner:

    def build(self, question, intent):

        if intent == "investment":

            return (
                question
                + " guidance outlook risks "
                + "capital allocation "
                + "revenue growth "
                + "free cash flow "
                + "buyback "
                + "management outlook"
            )

        elif intent == "financial":

            return (
                question
                + " revenue margin eps "
                + "cash flow "
                + "financial performance"
            )

        elif intent == "valuation":

            return (
                question
                + " dcf valuation "
                + "wacc intrinsic value "
                + "enterprise value"
            )

        elif intent == "transcript":

            return (
                question
                + " management commentary "
                + "CEO CFO "
                + "earnings call"
            )

        return question