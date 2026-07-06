class InvestmentThesisGenerator:
  def generate_thesis(self,generated_analysis,context.valuation_results):
    enterprise_value=context.valuation_results["enterprise_value"]
    equity_value=context.valuation_results["equity_value"]
    bullish_signals=[
      "Strong AI demand trends",
      "Enterprise adoption expansion",
      "Margin improvement"
    ]
    bearish_signals=[
       "Macroeconomic uncertainty",
       "Regulatory risks",
       "Competitive pressure"
    ]
    valuation_summary = (
            f"DCF valuation estimates equity value "
            f"at ${equity_value:,.0f}"
        )
    recommendation="Moderately Bullish"

    return {
         "bullish_signals": bullish_signals,
         "bearish_signals": bearish_signals,
         "valuation_summary": valuation_summary,
         "recommendation": recommendation
    }

