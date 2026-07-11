"""
planner.py

Public entry point used by ResearchAgent. Kept as a thin facade
(same name/shape as the original rule-based Planner) so nothing
downstream needs to change how it imports/uses the planner --
only its internals changed, from a flat keyword table to an LLM
planner with a deterministic rule-based safety net.
"""

from typing import List

from .llm_planner import LLMPlanner


class Planner:
    """
    Decides which tools are required to answer a user's question.

    Delegates to LLMPlanner, which itself falls back to
    `apply_fallback_rules` (see planner_rules.py) whenever the LLM's
    output can't be trusted. Callers only ever see a clean
    List[str] of tool names.
    """

    def __init__(self):
        self.llm_planner = LLMPlanner()

    def create_plan(self, question: str) -> List[str]:
        return self.llm_planner.create_plan(question)
