"""
planner_rules.py

Deterministic, rule-based fallback plan used when the LLM planner is
unavailable or returns something unusable (bad JSON, unknown tool
names, empty plan). Tool names were updated to match the
consolidated tool set in app/tools/ (the old rules referenced
market_tool/financial_tool/company_tool separately, and a
"thesis_tool" that never existed).

`report_tool` and `evaluation_tool` are intentionally NOT listed here
-- ResearchAgent always appends them at the end of every plan, so
rules only need to describe which *evidence-gathering* tools apply.
"""

FALLBACK_RULES = [

    (
        ["compare", " vs ", " vs.", "versus", "better buy"],
        ["market_data_tool", "valuation_tool", "comparison_tool"]
    ),

    (
        ["valuation", "dcf", "intrinsic", "undervalued", "overvalued", "fair value"],
        ["market_data_tool", "valuation_tool"]
    ),

    (
        ["revenue", "cash flow", "balance sheet", "income statement", "margin", "eps", "roe"],
        ["market_data_tool"]
    ),

    (
        ["earnings", "transcript", "conference call", "management", "guidance", "said", "mentioned"],
        ["rag_tool"]
    ),

    (
        ["sentiment", "tone", "positive", "negative", "optimistic", "cautious"],
        ["rag_tool", "sentiment_tool"]
    ),

    (
        ["buy", "sell", "invest", "recommend", "worth", "should i"],
        ["market_data_tool", "valuation_tool", "rag_tool", "sentiment_tool"]
    ),

]


def apply_fallback_rules(question: str):
    """Returns an ordered, de-duplicated tool list matching any rule that fires."""

    question = question.lower()
    selected = []

    for keywords, tools in FALLBACK_RULES:
        if any(keyword in question for keyword in keywords):
            for tool in tools:
                if tool not in selected:
                    selected.append(tool)

    if not selected:
        # Fully generic question: still ground the answer in real
        # financials rather than returning an empty plan.
        selected = ["market_data_tool"]

    return selected
