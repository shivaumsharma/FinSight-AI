from app.analysis.investment_thesis import InvestmentThesisGenerator
from .base_tool import BaseTool


class ThesisTool(BaseTool):

    name = "thesis_tool"

    description = "Generates the final investment thesis."

    def __init__(self):
        self.generator = InvestmentThesisGenerator()

    def run(
        self,
        generated_analysis,
        valuation_results,
        **kwargs
    ):

        return self.generator.generate_thesis(
            generated_analysis,
            valuation_results
        )