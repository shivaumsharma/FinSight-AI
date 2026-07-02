import streamlit as st
import pandas as pd
from app.rag.rag_pipeline import RAGPipeline
from app.rag.report_generator import ReportGenerator
from app.valuation.valuation_pipeline import ValuationPipeline
from app.data.financial_normalizer import (FinancialStatementNormaliser)
from app.data.market_data import (MarketDataLoader)
from app.analysis.investment_thesis import (InvestmentThesisGenerator)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------


if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False


st.set_page_config(
    page_title="Finsight AI",
    layout="wide"
)

DEFAULT_STATE={
    "valuation_results":None,
    "generated_response": None,
    "retrieved_chunks": None,
    "financial_df": None,
    "company_info": None,
    "market_cap": None,
    "beta": None,
    "transcript_text": None,
}

for key,value in DEFAULT_STATE.items():
    st.session_state.setdefault(key,value)
st.title("Finsight AI")
st.subheader(
    "AI-Powered Financial Research Platform"
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Transcript Settings")

ticker = st.sidebar.text_input(
    "Ticker Symbol",
    value="AAPL"
)

TRANSCRIPT_MAPPING = {
    "AAPL": "app/data/apple_q2.txt",
    "NVDA": "app/data/nvda_q2.txt",
    "MSFT": "app/data/msft_q2.txt"
}

transcript_path = TRANSCRIPT_MAPPING.get(
    ticker.upper(),
    "app/data/apple_q2.txt"
)

st.sidebar.success(
    f"Loaded transcript for {ticker.upper()}"
)

st.sidebar.title("Finsight AI")

st.sidebar.markdown(
    "AI-Powered Equity Research Platform"
)

st.sidebar.markdown("""
### Example Queries
- What did management say about AI demand?
- What drove margin expansion?
- How is China performing?
""")

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------

query = st.text_input(
    "Ask Financial Question",
    placeholder="What did management say about AI demand?"
)

# ---------------------------------------------------
# LOAD MARKET DATA
# ---------------------------------------------------

loader = MarketDataLoader(ticker)

income_stmt = loader.get_income_statement()

balance_sheet = loader.get_balance_sheet()

cash_flow = loader.get_cash_flow()

company_info = loader.get_company_info()

market_cap = company_info["market_cap"]

beta = 1.2

# ---------------------------------------------------
# NORMALIZE FINANCIALS
# ---------------------------------------------------

normaliser = FinancialStatementNormaliser(
    income_stmt,
    balance_sheet,
    cash_flow
)

financial_df = normaliser.normalise()

# ---------------------------------------------------
# VALIDATE REQUIRED COLUMNS
# ---------------------------------------------------

required_columns =[
    "revenue",
    "ebit",
    "tax_expense",
    "pretax_income",
    "depreciation",
    "capex",
    "current_assets",
    "current_liabilities",
    "interest_expense",
    "total_debt",
    "cash",
    "shares_outstanding"
]

missing_columns =[
    col for col in required_columns
    if col not in financial_df.columns
]

if missing_columns:
    st.error(
        f"Missing financial metrics: {missing_columns}"
    )
    st.stop()

# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------
with open(transcript_path,"r",encoding="utf-8") as file:
        transcript_text = file.read()

        if not st.session_state.analysis_complete:
            st.text_area("Transcript Preview",transcript_text[:1000],height=300)
        else:
             with st.expander("View Transcript"):
                  st.text_area(
                       "TranscriptPreview",
                               transcript_text[:1000],
                               height=300
                    )
     

if st.button("Run RAG Pipeline"):

    # -----------------------------------------------
    # RAG PIPELINE
    # -----------------------------------------------
    st.session_state.analysis_complete = True

    with st.spinner("Processing Transcript..."):

        rag_pipeline = RAGPipeline()

        rag_pipeline.ingest_transcript(
            transcript_path
        )

        retrieved_chunks = (
            rag_pipeline.query_pipeline(
                query=query,
                n_results=3
            )
        )

        report_generator = ReportGenerator()

        generated_response = (
            report_generator.generate_response(
                query=query,
                retrieved_chunks=retrieved_chunks
            )
        )

    st.success("Semantic retrieval completed")

    # -----------------------------------------------
    # AI ANALYSIS
    # -----------------------------------------------

    st.subheader("AI Generated Financial Analysis")

    st.write(generated_response)

    st.markdown("---")

    # -----------------------------------------------
    # DCF VALUATION
    # -----------------------------------------------
    DCF_KEYWORDS=[
        "dcf",
        "valuation",
        "intrinsic",
        "fair value",
        "undervalued",
        "overvalued",
        "equity value",
        "enterprise value",
        "wacc",
        "fcff"
    ]
    query_lower=(query or "").lower()
    run_dcf=any(keyword in query.lower()for keyword in DCF_KEYWORDS)
    if run_dcf:
        
        with st.expander("DCF Valuation Analysis"):

            valuation_pipeline = ValuationPipeline(
                financial_df=financial_df,
                market_cap=market_cap,
                beta=beta
            )
            valuation_results = (valuation_pipeline.run_valuation())
            st.metric(
                "Estimated Equity Value",
                f"${valuation_results['equity_value']:,.0f}"
            )
            st.metric(
                "Estimated Enterprise Value",
                f"${valuation_results['enterprise_value']:,.0f}"
            )
        
        
    # -----------------------------------------------
    # FCFF FORECASTS
    # -----------------------------------------------

        with st.expander("Projected FCFF"):

            fcff_df = valuation_results["fcff_forecasts"]

            chart_df=pd.DataFrame({"Year":[f"Year {i}"
            for i in range(1,len(fcff_df)+1)],
            "FCFF":fcff_df["forecast_fcff"].values
            })

            for year, fcff in enumerate(fcff_df["forecast_fcff"],start=1):
                st.write(f"Year {year}: ${fcff:,.0f}")
                st.line_chart(chart_df.set_index("Year"))

            st.markdown("---")

        # -----------------------------------------------
        # INVESTMENT THESIS
        # -----------------------------------------------

            thesis_generator = (InvestmentThesisGenerator())

            investment_thesis = (
                thesis_generator.generate_thesis(
                    generated_response,
                     valuation_results
                ))

            st.subheader("Investment Thesis")

            st.markdown("### Bullish Signals")

            for signal in investment_thesis["bullish_signals"]:
                st.write(f"✅ {signal}")

            st.markdown("### Bearish Signals")

            for signal in investment_thesis["bearish_signals" ]:
                st.write(f"⚠️ {signal}")

            st.markdown("### Valuation Summary")

            st.write(investment_thesis["valuation_summary"])

            st.markdown("### Recommendation")

            st.success(
                investment_thesis[
                    "recommendation"
                ]
            )

            st.markdown("---")

            # -----------------------------------------------
            # RETRIEVED CHUNKS
            # -----------------------------------------------

            for i, chunk in enumerate(
                retrieved_chunks
            ):

                st.markdown(
                    f"### Chunk {i+1}"
                )

                st.write(chunk)
        
        # -----------------------------------------------
        # Sensitivity Analysis
        # -----------------------------------------------
        
        with st.expander("DCF Sensitivity Analysis"):

            st.dataframe(valuation_results.keys())

            st.session_state["generated_response"] = generated_response
            st.session_state["valuation_results"] = valuation_results
            st.session_state["retrieved_chunks"] = retrieved_chunks
            st.session_state["financial_df"] = financial_df
            st.session_state["company_info"] = company_info
            st.session_state["market_cap"] = market_cap
            st.session_state["beta"] = beta

            st.write(type(valuation_results["sensitivity_analysis"]))
            st.write(valuation_results["sensitivity_analysis"])


        # ---------------------------------------------------
        # FINANCIAL DATA
        # ---------------------------------------------------

        with st.expander("View Financial Data"):
                st.dataframe(financial_df)

            # ---------------------------------------------------
            # FOOTER
            # ---------------------------------------------------


st.caption("Built using RAG, ChromaDB, FinBERT and DCF valuation models")