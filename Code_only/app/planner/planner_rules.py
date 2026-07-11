TOOL_RULES = [

    (
        ["market cap", "sector", "industry", "company"],
        ["company_tool"]
    ),

    (
        ["revenue", "cash flow", "balance sheet", "income statement"],
        ["financial_tool"]
    ),

    (
        ["valuation", "dcf", "intrinsic", "undervalued", "overvalued"],
        ["valuation_tool"]
    ),

    (
        ["earnings", "transcript", "conference call"],
        ["rag_tool"]
    ),

    (
        ["sentiment", "positive", "negative"],
        ["sentiment_tool"]
    ),

    (
        ["buy", "sell", "invest", "recommendation"],
        [
            "company_tool",
            "financial_tool",
            "valuation_tool",
            "rag_tool",
            # "thesis_tool"
        ]
    )

]