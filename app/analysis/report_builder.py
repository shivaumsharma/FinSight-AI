from app.core.research_context import ResearchContext


class ResearchSummaryBuilder:

    def build(self, context: ResearchContext) -> str:

        info = context.company_info

        financial = context.financial_summary or {}
        valuation = context.valuation_summary or {}
        sentiment = context.sentiment_summary or {}

        lines = []

        lines.append("=" * 60)
        lines.append("COMPANY")
        lines.append("=" * 60)

        lines.append(f"Name: {info.get('company_name','Unknown')}")
        lines.append(f"Sector: {info.get('sector','Unknown')}")
        lines.append(f"Industry: {info.get('industry','Unknown')}")
        lines.append("")

        lines.append("=" * 60)
        lines.append("FINANCIAL INTELLIGENCE")
        lines.append("=" * 60)

        # ------------------------------
        # Growth
        # ------------------------------
        lines.append("Growth")
        growth_metrics = [
            "Revenue Growth (%)",
            "Revenue CAGR (%)",
            "EBIT Growth (%)",
            "Net Income Growth (%)",
            "FCF Growth (%)"
        ]
        for metric in growth_metrics:
            if metric in financial:
                lines.append(
                    f"• {metric}: {financial[metric]}"
                )
        lines.append("")
        # ------------------------------
        # Profitability
        # ------------------------------
        lines.append("Profitability")
        profit_metrics = [
            "Operating Margin",
            "Net Margin",
            "ROE",
            "EPS"
        ]
        for metric in profit_metrics:
            if metric in financial:
                lines.append(
                    f"• {metric}: {financial[metric]}"
                )
        lines.append("")
        # ------------------------------
        # Balance Sheet
        # ------------------------------
        lines.append("Balance Sheet")
        balance_metrics = [
            "Debt to Equity"
        ]

        for metric in balance_metrics:
            if metric in financial:
                lines.append(
                    f"• {metric}: {financial[metric]}"
                )
        lines.append("")
        # ------------------------------
        # AI Interpretation
        # ------------------------------

        lines.append("Financial Interpretation")

        interpretation_metrics = [
            "Revenue Analysis",
            "Profitability",
            "Cash Flow",
            "Leverage",
            "ROE Analysis"
        ]
        for metric in interpretation_metrics:
            if metric in financial:
                lines.append(f"• {metric}: {financial[metric]}")

        lines.append("")

        lines.append("=" * 60)
        lines.append("VALUATION")
        lines.append("=" * 60)

        if valuation:
            for k, v in valuation.items():
                lines.append(f"{k}: {v}")
        else:
            lines.append("Unavailable")

        lines.append("")

        lines.append("=" * 60)
        lines.append("SENTIMENT")
        lines.append("=" * 60)

        if sentiment:
            for k, v in sentiment.items():
                lines.append(f"{k}: {v}")
        else:
            lines.append("Unavailable")

        lines.append("")

        lines.append("=" * 60)
        lines.append("EARNINGS CALL EVIDENCE")
        lines.append("=" * 60)

        if context.retrieved_chunks:
            for i, chunk in enumerate(context.retrieved_chunks, 1):
                lines.append(f"Evidence {i}")
                lines.append(
                    f"Speaker: {chunk['metadata'].get('speaker','Unknown')}"
                )
                lines.append(
                    f"Section: {chunk['metadata'].get('section','General')}"
                )
                lines.append(chunk["text"])
                lines.append("")
        else:
            lines.append("No evidence retrieved.")

        lines.append("")
        lines.append("=" * 60)
        lines.append("KNOWN LIMITATIONS")
        lines.append("=" * 60)

        if not financial:
            lines.append("- Financial metrics unavailable.")

        if not valuation:
            lines.append("- Valuation unavailable.")

        if not sentiment:
            lines.append("- Sentiment unavailable.")

        if not context.retrieved_chunks:
            lines.append("- Earnings call evidence unavailable.")
        
        for i, item in enumerate(lines):
            if not isinstance(item, str):
                print(i)
                print(type(item))
                print(item)

        return "\n".join(lines)