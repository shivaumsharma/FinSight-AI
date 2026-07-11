"""
llm_planner.py

The agentic core of the redesign: an LLM decides which tools to run
for a given question, instead of a fixed keyword->tool table.

    "Summarize Apple's earnings call"       -> ["rag_tool"]
    "Calculate Apple's intrinsic value"     -> ["market_data_tool", "valuation_tool"]
    "Compare Apple and Microsoft"           -> ["market_data_tool", "valuation_tool", "comparison_tool"]

Design choices
--------------
- No agent framework (LangGraph/CrewAI/AutoGen/Semantic Kernel).
  This is ~60 lines of plain Python: build a prompt, call a model,
  parse JSON, validate against the tool registry, fall back to
  rules on any failure. That's the entire "framework."
- Reuses the same local Qwen2.5-1.5B-Instruct model already used for
  report writing (via `get_shared_generator`), so the redesign adds
  zero new external dependencies or API keys.
- The planner is asked ONLY to choose tools from a fixed, known
  vocabulary (the registry's tool names). It is never asked to
  invent tool names, arguments, or company tickers -- ticker/company
  resolution is handled separately and deterministically
  (see app/core/ticker_resolver.py) because that is a closed-set
  lookup problem, not a reasoning problem, and LLMs are an
  unnecessarily unreliable way to spell "AAPL".
- Every failure mode (model not available, malformed JSON, unknown
  tool name, empty plan) degrades to the deterministic
  `FALLBACK_RULES` planner rather than raising -- an agentic system
  in a research/finance context should never go fully silent just
  because a single LLM call had a bad day.
"""

import json
import re
from typing import List

from .planner_rules import apply_fallback_rules

# Tools the LLM planner is allowed to choose. report_tool and
# evaluation_tool are deliberately excluded: the ResearchAgent always
# appends them itself, so the planner never wastes a token deciding
# whether to "write the report" -- of course it should.
PLANNABLE_TOOLS = {
    "market_data_tool": "Fetches company profile, financial statements and normalizes them. Needed for financial-metric, valuation, and investment questions.",
    "valuation_tool": "Runs a DCF valuation (WACC, enterprise value, intrinsic value). Needed for valuation/intrinsic-value/undervalued questions.",
    "rag_tool": "Retrieves relevant earnings-call transcript evidence with citations. Needed for questions about management commentary, guidance, or the earnings call.",
    "sentiment_tool": "Runs FinBERT sentiment analysis on transcript evidence. Needed for tone/sentiment questions.",
    "comparison_tool": "Compares two companies side by side. Needed ONLY when the question names two companies to compare.",
}

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class LLMPlanner:
    """
    Rule-informed LLM planner: asks a small local LLM to choose an
    ordered subset of PLANNABLE_TOOLS as strict JSON, falling back to
    `apply_fallback_rules` whenever the model's output can't be
    trusted.
    """

    def __init__(self, generator=None):
        # Injected for testability; defaults to the shared singleton
        # so no extra copy of the model is loaded in production.
        self._generator = generator

    @property
    def generator(self):
        if self._generator is None:
            from app.core.llm_provider import get_shared_generator
            self._generator = get_shared_generator()
        return self._generator

    def create_plan(self, question: str) -> List[str]:

        try:
            raw = self.generator.generate(self._build_prompt(question), max_new_tokens=60)
            llm_plan = self._parse_plan(raw)
        except Exception:
            llm_plan = []

        # IMPORTANT: the deterministic keyword rules are now applied
        # unconditionally and unioned with whatever the LLM chose --
        # not just used as a fallback when the LLM output is empty
        # or malformed.
        #
        # Previously, a small local model could return a perfectly
        # *valid* but *wrong* plan (e.g. only ["rag_tool"] for an
        # investment question), and because that plan was non-empty
        # and used only known tool names, `apply_fallback_rules`
        # never got a chance to run. That silently starved
        # market_data_tool / valuation_tool / sentiment_tool of
        # ever executing on investment-style questions, which is
        # why company/financials/valuation/sentiment kept coming
        # back "Unknown" / "Unavailable" even though the rules for
        # those questions were defined correctly.
        #
        # Taking the union means: the LLM can still *add* extra
        # tools it judges relevant, but it can no longer *omit*
        # tools that the deterministic rules say the question
        # clearly needs.
        rule_plan = apply_fallback_rules(question)

        plan = list(llm_plan)
        for tool in rule_plan:
            if tool not in plan:
                plan.append(tool)

        return plan

    # -----------------------------------------------------------

    def _build_prompt(self, question: str) -> str:

        tool_list = "\n".join(
            f'- "{name}": {desc}' for name, desc in PLANNABLE_TOOLS.items()
        )

        return f"""You are a planning module for a financial research agent.

Available tools:
{tool_list}

You are ONLY a tool planner.

Your job is to MINIMIZE the number of tools used.

Never choose unnecessary tools.

Rules:

- Questions about earnings calls or management commentary:
  Use ONLY ["rag_tool"]

- Questions about sentiment:
  Use ONLY ["rag_tool", "sentiment_tool"]

- Questions about valuation, intrinsic value, DCF, fair value:
  Use ONLY ["market_data_tool", "valuation_tool"]

- Questions about company financial metrics:
  Use ONLY ["market_data_tool"]

- Questions comparing two companies:
  Use ONLY ["market_data_tool", "valuation_tool", "comparison_tool"]

- Questions asking whether to invest/buy/sell/hold, or if a stock is
  "worth it":
  Use ["market_data_tool", "valuation_tool", "rag_tool", "sentiment_tool"]

Never include tools "just in case."

Return ONLY JSON.

Example:

Question:
"What did management say about AI?"

{"tools":["rag_tool"]}

Question:
"What is Apple's intrinsic value?"

{"tools":["market_data_tool","valuation_tool"]}

Question:
"Should I invest in Apple?"

{"tools":["market_data_tool","valuation_tool","rag_tool","sentiment_tool"]}

Respond with ONLY a JSON object of the form:
{{"tools": ["tool_name_1", "tool_name_2"]}}

No explanation. No markdown. JSON only.

Question: "{question}"
JSON:"""

    def _parse_plan(self, raw_output: str) -> List[str]:

        match = _JSON_BLOCK.search(raw_output)
        if not match:
            return []

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

        tools = parsed.get("tools", [])
        if not isinstance(tools, list):
            return []

        plan = []
        for tool in tools:
            if isinstance(tool, str) and tool in PLANNABLE_TOOLS and tool not in plan:
                plan.append(tool)

        return plan