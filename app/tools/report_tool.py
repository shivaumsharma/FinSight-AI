
"""
report_tool.py

Terminal tool of almost every plan. Consumes whatever the earlier
tools populated on the context (financials, valuation, sentiment,
RAG evidence -- any subset, including none) and produces the final
natural-language equity research report.

Reuses, unchanged:
 - ResearchSummaryBuilder  (app/analysis/report_builder.py)
 - PromptBuilder           (app/core/prompt_builder.py)
 - ReportGenerator         (app/rag/report_generator.py)

This is exactly what ReportGenerationEngine + the tail of
EvidenceBuilder used to do inline inside streamlit_app.py. Wrapping
it as a tool means the agent (not the UI script) owns the decision
of when to call it, and it can be reused identically by
ComparisonTool for each side of a two-company comparison.
"""

import re

from app.core.research_context import ResearchContext
from app.analysis.report_builder import ResearchSummaryBuilder
from app.core.prompt_builder import PromptBuilder
from app.core.llm_provider import get_shared_generator
from .base_tool import BaseTool


# The 1.5B local model reliably fills in the 5-section template, but
# does not reliably *stop* once it's done -- with a larger token
# budget it tends to drift into meta-commentary about the task itself
# ("Please provide feedback on how accurately you captured...",
# "Assessment: ...") and, if it keeps going past that, into
# degenerate repeated filler/emoji. None of that is part of the
# report, but it was getting fed into grounding/citation scoring as
# if it were, which dragged both scores toward 0 regardless of how
# good the actual report was. _trim_runaway_output cuts the
# generated text back down to just the report once it detects the
# model has moved past writing it.
_REQUIRED_SECTIONS = [
    "Executive Summary",
    "Bull Case",
    "Bear Case",
    "Financial Outlook",
    "Investment Recommendation",
]

_DRIFT_MARKERS = (
    "please provide", "assessment:", "response:", "suggestions:",
    "let me know", "your response should", "for example:",
    "thank you", "feedback on", "your insights will help",
)


def _looks_like_drift(line: str) -> bool:
    lowered = line.lower()
    if any(marker in lowered for marker in _DRIFT_MARKERS):
        return True
    # Degenerate emoji/symbol runs (each character here is well past
    # the ASCII/basic-punctuation range).
    non_ascii = sum(1 for ch in line if ord(ch) > 0x2500)
    if non_ascii > 5:
        return True
    return False


def _trim_runaway_output(text: str) -> str:

    last_heading_end = -1

    for heading in _REQUIRED_SECTIONS:
        matches = list(re.finditer(
            rf"^#*\s*{re.escape(heading)}", text, re.IGNORECASE | re.MULTILINE
        ))
        if matches:
            end = matches[-1].start()
            if end > last_heading_end:
                last_heading_end = end

    if last_heading_end == -1:
        # No recognizable heading at all -- nothing safe to anchor
        # on, leave the text alone rather than guess.
        return text.strip()

    head, tail = text[:last_heading_end], text[last_heading_end:]

    kept_lines = []
    for line in tail.split("\n"):
        if line.strip() and _looks_like_drift(line.strip()):
            break
        kept_lines.append(line)

    return (head + "\n".join(kept_lines)).strip()


class ReportTool(BaseTool):

    name = "report_tool"
    description = (
        "Synthesizes all previously gathered evidence (financials, valuation, "
        "sentiment, transcript citations) into a final structured equity "
        "research report. Should almost always be the last tool in any plan."
    )

    def run(self, context: ResearchContext) -> ResearchContext:

        context.research_summary = ResearchSummaryBuilder().build(context)

        prompt = PromptBuilder().build(context)

        raw_answer = get_shared_generator().generate(prompt)

        context.generated_answer = _trim_runaway_output(raw_answer)

        context.record_tool(self.name)

        return context