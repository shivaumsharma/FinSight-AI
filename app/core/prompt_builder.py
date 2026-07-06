"""
prompt_builder.py

Builds a clean prompt for the language model.

Responsibilities
----------------
✓ Read ResearchContext
✓ Format important information
✓ Construct the final prompt

Does NOT
---------
✗ Call the LLM
✗ Fetch data
✗ Run valuation
✗ Perform retrieval
"""

from app.core.research_context import ResearchContext


class PromptBuilder:

    def build(self, context: ResearchContext) -> str:

        sections = [
            self._system_prompt(),
            self._question(context),
            self._company(context),
            self._valuation(context),
            self._sentiment(context),
            self._evidence(context),
            self._instructions(),
        ]

        return "\n\n".join(sections)

    # ==========================================================
    # System Prompt
    # ==========================================================

    def _system_prompt(self):

        return """
You are a CFA Level III Equity Research Analyst.

You analyze companies using ONLY the supplied information.

Never invent facts.

If information is missing, explicitly state that.

Write concise, professional equity research reports.
"""

    # ==========================================================
    # User Question
    # ==========================================================

    def _question(self, context):

        return f"""
QUESTION

{context.question}
"""

    # ==========================================================
    # Company
    # ==========================================================

    def _company(self, context):

        info = context.company_info or {}

        return f"""
COMPANY

Name: {info.get("longName", "Unknown")}

Sector: {info.get("sector", "Unknown")}

Industry: {info.get("industry", "Unknown")}

Market Cap: {context.market_cap}

Beta: {context.beta}
"""

    # ==========================================================
    # Valuation
    # ==========================================================

    def _valuation(self, context):

        valuation = context.valuation_results or {}

        return f"""
VALUATION

Intrinsic Value:
{valuation.get("intrinsic_value", "Unavailable")}

Enterprise Value:
{context.enterprise_value}

Equity Value:
{context.equity_value}
"""

    # ==========================================================
    # Sentiment
    # ==========================================================

    def _sentiment(self, context):

        sentiment = context.sentiment or {}

        return f"""
MARKET SENTIMENT

Overall Sentiment:
{sentiment}
"""

    # ==========================================================
    # Evidence
    # ==========================================================

    def _evidence(self, context):

        if not context.retrieved_chunks:

            evidence = "No evidence retrieved."

        else:

            # Use only the most relevant chunks
            evidence = "\n\n".join(
                context.retrieved_chunks[:2]
            )

        return f"""
EARNINGS CALL EVIDENCE

{evidence}
"""

    # ==========================================================
    # Instructions
    # ==========================================================

    def _instructions(self):

        return """
TASK

Based ONLY on the information above, answer the investment question.

Your report should include:

Executive Summary

Bull Case

Bear Case

Financial Outlook

Major Risks

Investment Recommendation (Buy, Hold or Sell)

Confidence Score (0-100)

Do not repeat these instructions.

Do not invent information.

Keep the response under 400 words.
"""