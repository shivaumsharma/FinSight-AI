"""
streamlit_app.py

UI entry point. Previously this file called EvidenceBuilder and
ReportGenerationEngine directly and hard-coded a transcript-path
lookup that pointed at files that don't exist
("app/data/apple_q2.txt" vs the real "app/data/transcripts/apple_q2.txt").

It now delegates all orchestration to ResearchAgent, which:
 - resolves ticker(s) from the free-text question,
 - asks the LLM Planner which tools are needed,
 - executes them,
 - always finishes with report_tool + evaluation_tool.

The UI's only job is to render whatever ends up on the returned
ResearchContext -- it does not know or care which tools ran.
"""

import time

import streamlit as st

from app.agents.research_agent import ResearchAgent
from app.core.logger import ResearchLogger

st.set_page_config(page_title="Finsight AI", layout="wide")

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

st.title("Finsight AI")
st.subheader("Agentic Financial Research Platform")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")

st.sidebar.markdown("""
### Example Queries
- What did management say about AI demand?
- Calculate Apple's intrinsic value
- Compare Apple and Microsoft
- Should I invest in Apple?
""")

# ---------------------------------------------------
# USER INPUT
# ---------------------------------------------------

query = st.text_input(
    "Ask Financial Question",
    placeholder="What did management say about AI demand?",
)

# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------

if st.button("Run Research Agent") and query:

    start = time.time()

    with st.spinner("Planning and executing tools..."):
        context = ResearchAgent().run(question=query, ticker=ticker)

    end = time.time()

    ResearchLogger().save(context)

    st.session_state.analysis_complete = True

    with st.expander(f"Agent Plan (mode: {context.mode})", expanded=False):
        st.write(context.metadata.get("plan", []))
        st.caption(f"Tools actually executed: {context.tool_trace}")

    if context.mode == "comparison":
        st.info(f"Comparing {context.ticker} vs {context.metadata.get('peer_ticker')}")

    with st.expander("Research Summary", expanded=False):
        st.text(context.research_summary)

    st.markdown("## AI Investment Analysis")
    st.write(context.generated_answer)

    if context.citations:
        with st.expander("Retrieved Evidence"):
            for citation in context.citations:
                st.markdown(f"### {citation['id']}")
                st.write(citation["text"])

    if context.valuation_results:
        with st.expander("Valuation Detail"):
            st.json(context.valuation_summary)

    evaluation = context.evaluation or {}

    if evaluation:
        st.subheader("AI Evaluation")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Overall Score", f"{evaluation.get('overall_score', 0):.1f}")
        with col2:
            st.metric("Grounding", f"{evaluation.get('grounding_score', 0):.1f}%")
        with col3:
            st.metric("Retrieval", f"{evaluation.get('retrieval_score', 0):.1f}%")

        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Citation Coverage", f"{evaluation.get('citation_score', 0):.1f}%")
        with col5:
            st.metric("Completeness", f"{evaluation.get('completeness_score', 0):.1f}%")
        with col6:
            st.metric("Latency", f"{end - start:.2f}s")

st.caption("Built using an LLM Planner + RAG, ChromaDB, FinBERT and DCF valuation tools")
