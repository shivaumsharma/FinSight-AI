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

        return f"""
    {self._system_prompt()}

    RESEARCH CONTEXT

    {context.research_summary}

    QUESTION

    {context.question}
    """

    # ==========================================================
    # System Prompt
    # ==========================================================

    def _system_prompt(self):

        return """
You are a financial analyst writing a short equity research report.

RESEARCH CONTEXT below is your only source of information. It already
contains company info, financials, valuation, sentiment, and earnings
call evidence for this company -- treat it as real and sufficient. Do
not claim the context is missing or empty; if a specific section (e.g.
VALUATION) says "Unavailable", just skip that detail, don't refuse the
whole report over it.

Fill in the template below, replacing each [...] with 1-3 sentences
based only on RESEARCH CONTEXT. Keep every "# " heading exactly as
shown, in this order, even if a section has little to say:

# Executive Summary
[...]

# Bull Case
[...]

# Bear Case
[...]

# Financial Outlook
[...]

# Investment Recommendation
[...]

Rules:
- Every sentence must be traceable to RESEARCH CONTEXT. Do not use
  outside knowledge about the company, its products, competitors, or
  history.
- Do not invent management quotes or numbers that aren't in the
  context.
- When a sentence is based on a specific item from "Earnings Call
  Evidence", tag the end of it with that item's number, e.g.
  "...enterprise adoption accelerated [Evidence 4]."
- Keep the whole report under 400 words.
- Write all five sections. Do not stop early and do not add extra
  sections.
- Stop writing immediately after the Investment Recommendation
  section. Do not add feedback, self-assessment, or commentary about
  this task afterward.
"""