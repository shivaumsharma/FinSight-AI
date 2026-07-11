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


class ReportGenerationEngine:

    def __init__(self):

        self.generator = ReportGenerator()

    # def run(self,context: ResearchContext) -> ResearchContext:

    #   builder = PromptBuilder()
    #   prompt = builder.build(context)
    #   context.generated_answer = self.generator.generate(prompt)
    def run(self, context: ResearchContext) -> ResearchContext:

        builder = PromptBuilder()

        prompt = builder.build(context)

        print("\n" + "=" * 100)
        print("FINAL PROMPT SENT TO LLM")
        print("=" * 100)
        print(prompt)
        print("=" * 100 + "\n")

        context.generated_answer = self.generator.generate(prompt)

        return context
      
        
        

   