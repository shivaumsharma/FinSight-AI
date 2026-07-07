"""
report_sections.py

Generates each section of the final equity research report
individually using the LLM.
"""


class ReportSectionGenerator:

    def __init__(self, llm):
        self.llm = llm

    # =====================================================
    # Executive Summary
    # =====================================================

    def generate_summary(self, prompt: str) -> str:
        return self.llm.generate(prompt)

    # =====================================================
    # Bull Case
    # =====================================================

    def generate_bull_case(self, prompt: str) -> str:
        return self.llm.generate(prompt)

    # =====================================================
    # Bear Case
    # =====================================================

    def generate_bear_case(self, prompt: str) -> str:
        return self.llm.generate(prompt)

    # =====================================================
    # Financial Outlook
    # =====================================================

    def generate_financial_outlook(self, prompt: str) -> str:
        return self.llm.generate(prompt)

    # =====================================================
    # Recommendation
    # =====================================================

    def generate_recommendation(self, prompt: str) -> str:
        return self.llm.generate(prompt)

    # =====================================================
    # Combine Final Report
    # =====================================================

    def combine_sections(
        self,
        summary: str,
        bull: str,
        bear: str,
        financial: str,
        recommendation: str,
    ) -> str:

        return f"""
# Executive Summary

{summary}

# Bull Case

{bull}

# Bear Case

{bear}

# Financial Outlook

{financial}

# Investment Recommendation

{recommendation}
""".strip()