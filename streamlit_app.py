import streamlit as st
from app.rag.rag_pipeline import (RAGPipeline)
from app.rag.report_generator import(ReportGenerator)
from app.valuation.valuation_pipeline import (ValuationPipeline)
import pandas as pd 
from app.data.financial_normalizer import FinancialStatementNormaliser

st.set_page_config(page_title="Finsight AI",layout="wide")

st.title("Finsight AI")
st.subheader("AI-Powered Financial Research Platform")

st.sidebar.header("Transcript Settings")

transcript_path=st.sidebar.text_input("Transcript Path",value="app/data/apple_q2.txt")

query=st.text_input("Ask Financial Question",placeholder="What did management say about AI demand?")

financial_df = pd.DataFrame({

    "revenue": [
        365817000000,
        394328000000,
        383285000000
    ],

    "ebit": [
        119437000000,
        130541000000,
        123216000000
    ],

    "tax_expense": [
        19000,
        21000,
        20000
    ],

    "pretax_income": [
        100000,
        110000,
        105000
    ],

    "depreciation": [
        11000,
        12000,
        12500
    ],

    "capex": [
        9500,
        10700,
        11500
    ],

    "current_assets": [
        135000,
        143000,
        148000
    ],

    "current_liabilities": [
        95000,
        98000,
        102000
    ],

    "interest_expense": [
        2800,
        3000,
        3200
    ],

    "total_debt": [
        120000,
        118000,
        117000
    ],

    "cash": [
        62000,
        67000,
        71000
    ],

    "shares_outstanding": [
        15500000000,
        15400000000,
        15300000000
    ]

})
market_cap = 3200000000000
beta = 1.2


if st.button("Run RAG Pipeline"):
  with st.spinner("Processing Transcript..."):
    rag_pipeline=(RAGPipeline())
    rag_pipeline.ingest_transcript(transcript_path)
    retrieved_chunks=(rag_pipeline.query_pipeline(query=query,n_results=3))
    report_generator=ReportGenerator()
    generated_response=(report_generator.generate_response(query=query,retrieved_chunks=retrieved_chunks))
  st.success("Semantic retrieval completed")
  st.subheader("Retrieved Context")
  st.metric("Retrieved Chunks",len(retrieved_chunks))
  st.subheader("AI Generated Financial Analysis")
  st.write(generated_response)
  st.markdown("---")
  st.subheader("DCF Valuation Analysis")
  valuation_pipeline=ValuationPipeline(financial_df=financial_df,market_cap=market_cap,beta=beta)
  valuation_results=valuation_pipeline.run_valuation()
  st.metric("Estimated Equity Value",f"${valuation_results['equity_value']:,.0f}")
  st.metric("Estimated Enterprise Value ",f"${valuation_results['enterprise_value']:,.0f}")
  st.subheader("Projected FCFF")
  fcff_df = valuation_results["fcff_forecasts"]

  for year, fcff in enumerate(fcff_df["forecast_fcff"],start=1):
    st.write(f"Year {year}: ${fcff:,.0f}")

  for i,chunk in enumerate(retrieved_chunks):
    st.markdown(f"### Chunk {i+1}")
    st.write(chunk)

st.sidebar.title("Finsight AI")
st.sidebar.markdown("AI-Powered Equity Research Platform")
col1,col2=st.columns(2)
st.markdown("---")
st.caption("Built using RAG,ChromaDB,Finbert and DCF valuation models")
st.sidebar.markdown("""
### Example Queries
- What did management say about AI demand?
- What drove margin expansion?
- How is China performing?
""")
with open(transcript_path, "r", encoding="utf-8") as file:
     transcript_text=file.read()

st.text_area(
    "Transcript Preview",
    transcript_text[:1000],
    height=300
)

with st.expander("View Financial Data"):
    st.dataframe(financial_df)