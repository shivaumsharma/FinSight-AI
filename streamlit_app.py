import time
from app.core.research_context import ResearchContext
from app.core.ResearchPipeline import EvidenceBuilder
from app.core.reasoning_engine import ReportGenerationEngine
from app.evaluation.evaluation_engine import EvaluationEngine
from app.core.logger import ResearchLogger
import streamlit as st
import pandas as pd


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------


if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False


st.set_page_config(
    page_title="Finsight AI",
    layout="wide"
)

DEFAULT_STATE = {

    "context.valuation_results":None,

    "generated_answer":None,

    "retrieved_chunks":None,

    "financial_df":None,

    "company_info":None,

    "market_cap":None,

    "beta":None,

    "transcript_text":None

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
     start = time.time()

     context = ResearchContext(ticker=ticker,question=query,transcript_path=transcript_path)

     context = EvidenceBuilder().build(context)

     with st.expander("Research Summary"):
        st.text(context.research_summary)
     context = ReportGenerationEngine().run(context)

     end = time.time()

     evaluation = EvaluationEngine().evaluate(context=context,
        generated_report=context.generated_answer,
        latency=end - start,
    )

     ResearchLogger().save(context)
     with st.expander("Pipeline Metrics"):
        st.markdown("## AI Investment Analysis")
        st.write(context.generated_answer)
     
     with st.expander("Retrieved Evidence"):
        for citation in context.citations:
            st.markdown(f"### {citation['id']}")
            st.write(citation["text"])

     st.subheader("AI Evaluation")
     col1,col2,col3=st.columns(3)
     with col1:
         st.metric("Overall Score",f"{evaluation.overall_score:.1f}")

     with col2:
         st.metric("Grounding",f"{evaluation.grounding_score:.1f}%")

     with col3:
         st.metric("Retrieval",f"{evaluation.retrieval_score:.1f}%")

     col4,col5,col6=st.columns(3)
     with col4:
         st.metric( "Citation Coverage",f"{evaluation.citation_score:.1f}%")

     with col5:
         st.metric("Completeness",f"{evaluation.completeness_score:.1f}%")

     with col6:
         st.metric("Latency",f"{evaluation.latency:.2f}s")
    #  with st.expander("Evaluation Details"):

        # st.write("Supported Claims:", evaluation.supported_claims)

        # st.write("Unsupported Claims:", evaluation.unsupported_claims)

        # st.write("Retrieved Chunks:", evaluation.retrieved_chunks)

        # st.write("Reranked Chunks:", evaluation.reranked_chunks)

        # st.write("Citations Used:", evaluation.citations_used)

        # st.write("Available Citations:", evaluation.citations_available)

        # st.write("Missing Sections:", evaluation.missing_sections)
   
st.caption("Built using RAG, ChromaDB, FinBERT and DCF valuation models")