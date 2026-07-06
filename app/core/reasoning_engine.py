"""
reasoning_engine.py

Consumes a fully-populated ResearchContext and generates
an investment analysis.

This class performs NO data collection.

It assumes the EvidenceBuilder has already completed.
"""
from app.core.prompt_builder import PromptBuilder
from app.core.research_context import ResearchContext
from app.rag.report_generator import ReportGenerator

class ReasoningEngine:

    def __init__(self):

        self.generator = ReportGenerator()

    def run(
        self,
        context: ResearchContext
    ) -> ResearchContext:

        prompt = PromptBuilder().build(context)

        answer = self.generator.generate(prompt)

        context.generated_answer = answer

        return context

   