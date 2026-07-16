"""
streamlit_app.py

UI entry point.

FinSight AI is an Autonomous Financial Intelligence Platform, not a
general-purpose financial chatbot. There is no ticker input box and no
default company -- the user asks a question naming a real, publicly
listed company, FinSight resolves the company itself (see
app/core/company_resolver.py), and if none is found, no analysis
begins at all. Validation happens before ResearchAgent.run() ever
touches yfinance/SEC/the LLM -- NoCompanyDetectedError and
TickerNotFoundError are the only two ways this can fail, and both are
caught here with a plain-language message; no internal exception or
raw data-provider error ever reaches the user.
"""

import time

import streamlit as st

from app.agents.research_agent import ResearchAgent, NoCompanyDetectedError
from app.core.logger import ResearchLogger
from app.data.market_data import TickerNotFoundError

st.set_page_config(page_title="Finsight AI", layout="wide")

RATING_COLORS = {
    "Buy": "green",
    "Hold": "orange",
    "Sell": "red",
    "Insufficient Data": "gray",
}

# ---------------------------------------------------
# INTRODUCTION
# ---------------------------------------------------

st.title("Finsight AI")
st.subheader("Autonomous Financial Intelligence Platform")

st.markdown("""
FinSight AI is an Autonomous Financial Intelligence Platform designed for
institutional-style equity research and investment analysis. It specializes in:

- Financial Statement Analysis
- Equity Research
- Intrinsic Valuation
- Earnings Call & Filing Analysis
- Market Intelligence
- Investment Thesis Generation

**FinSight AI is not a general-purpose financial chatbot.** Please include a
publicly listed company as part of your query.

Examples: *"Should I buy Apple?"* &nbsp;&middot;&nbsp; *"Analyze NVIDIA's financial health."*
&nbsp;&middot;&nbsp; *"Generate a report on Microsoft."* &nbsp;&middot;&nbsp; *"Compare Google and Amazon."*
""")

# ---------------------------------------------------
# USER INPUT -- single field, no ticker box
# ---------------------------------------------------

query = st.text_input(
    "Ask about a publicly listed company",
    placeholder="Should I buy Apple?",
)

# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------

if st.button("Run Research Agent") and query:

    start = time.time()

    try:
        with st.spinner("Planning and executing research..."):
            context = ResearchAgent().run(question=query)
    except NoCompanyDetectedError:
        st.error(
            "FinSight AI specializes in company and investment research. No "
            "publicly listed company was detected in your query. Please "
            "provide a company name to begin the analysis."
        )
        st.stop()
    except TickerNotFoundError as e:
        st.error(
            f"Couldn't find market data for '{e}'. The company may be "
            "delisted, foreign-listed, or the name didn't resolve to a "
            "real ticker -- please check the spelling and try again."
        )
        st.stop()
    except Exception as e:
        st.error(f"Something went wrong while researching this: {e}")
        st.stop()

    end = time.time()

    ResearchLogger().save(context)

    report_data = context.report_data or {}
    recommendation = report_data.get("recommendation", {})
    rating = recommendation.get("rating", "Insufficient Data")
    color = RATING_COLORS.get(rating, "gray")

    if context.mode == "comparison":
        st.info(f"Comparing {context.ticker} vs {context.metadata.get('peer_ticker')}")

    st.markdown("## Report Ready")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"### :{color}[{rating}]")
        st.caption(recommendation.get("basis", ""))
        if recommendation.get("confidence_flag"):
            st.warning(f"**Low-confidence signal:** {recommendation['confidence_flag']}")
    with col2:
        confidence = report_data.get("confidence_scores", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Score", confidence.get("Overall Score", "N/A"))
        c2.metric("Grounding", f"{confidence.get('Grounding (%)', 'N/A')}")
        c3.metric("Latency", f"{end - start:.1f}s")

    narrative = report_data.get("narrative", {})
    if narrative.get("Executive Summary"):
        with st.expander("Executive Summary preview", expanded=True):
            st.write(narrative["Executive Summary"])

    consensus = (report_data.get("institutional_consensus") or {}).get("recommendation_consensus")
    if consensus:
        with st.expander("Institutional Consensus Score", expanded=True):
            st.markdown(f"### Institutional Consensus Score\n# {consensus['score']}%\n**{consensus['label']}**")
            st.caption(consensus["methodology"])

            st.markdown("---")
            st.markdown("**Market Consensus**")
            for r in consensus["institutional_ratings"]:
                r_color = RATING_COLORS.get(r["rating"].title(), "gray")
                st.markdown(f"{r['firm']} &nbsp;&nbsp; :{r_color}[**{r['rating']}**]")

            st.markdown("---")
            st.markdown(f"**FinSight Recommendation**\n\n:{color}[**{consensus['finsight_rating']}**]")

            st.markdown("---")
            st.markdown(f"**Market Summary**\n\n{consensus['summary']}")

    news_sources = report_data.get("news_sources", {})
    with st.expander(
        f"News Sources Used in This Analysis ({news_sources.get('total_retrieved', 0)} retrieved, "
        f"{news_sources.get('total_selected', 0)} used)",
        expanded=False,
    ):
        if not news_sources.get("total_retrieved"):
            st.caption("No recent news coverage was found for this company.")
        else:
            st.caption(
                "Every article retrieved is shown here, not just the ones used in "
                "the analysis above -- so you can judge whether the selection looks "
                "reasonable, not only what the model chose to reference."
            )
            for article in news_sources["all_articles"]:
                tag = "✅ Used" if article["used_in_analysis"] else "⬜ Retrieved, not used"
                st.markdown(f"**{article['headline']}**  \n{article['source']} — {article['date']} — [{tag}]({article['url']})")

    if context.pdf_bytes:
        st.download_button(
            label="Download Full Research Report (PDF)",
            data=context.pdf_bytes,
            file_name=f"{context.ticker}_FinSight_Research_Report.pdf",
            mime="application/pdf",
        )

st.caption("Built using an LLM Planner + RAG over live SEC filings, ChromaDB, FinBERT and DCF valuation tools")
