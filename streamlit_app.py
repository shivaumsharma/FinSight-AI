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
from app.valuation.what_if_dcf import (
    compute_what_if, DEFAULT_TERMINAL_GROWTH_RATE, GROWTH_RATE_MIN, GROWTH_RATE_MAX,
    TERMINAL_GROWTH_MIN, TERMINAL_GROWTH_MAX, WACC_OFFSET,
)
from app.valuation.fcff_engine import FCFFEngine

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

with st.form("query_form"):
    query = st.text_input(
        "Ask about a publicly listed company",
        placeholder="Should I buy Apple?",
    )
    submitted = st.form_submit_button("Run Research Agent")

# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------

if submitted and query:

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
    valuation_results = context.valuation_results or {}

    # What-if sliders (below) need to survive future reruns triggered
    # by moving a slider -- Streamlit reruns the whole script on every
    # widget interaction, and a bare local variable wouldn't survive
    # that. financial_df/total_debt/cash/shares/current_price/raw_wacc
    # are cached here once (cheap: already-fetched data, no new
    # network/LLM calls) so each slider move only re-runs the cheap
    # DCF recompute in what_if_dcf.py, not the full pipeline.
    whatif = None
    financial_df = context.normalized_financials
    if valuation_results.get("dcf_available") and financial_df is not None and not financial_df.empty:
        try:
            shares = financial_df["shares_outstanding"].dropna().iloc[-1]
            debt = financial_df["total_debt"].dropna().iloc[-1]
            cash_bal = financial_df["cash"].dropna().iloc[-1]
            current_price = valuation_results.get("current_price")
            if shares and current_price:
                whatif = {
                    "financial_df": financial_df,
                    "total_debt": debt,
                    "cash": cash_bal,
                    "shares_outstanding": shares,
                    "current_price": current_price,
                    "raw_wacc": valuation_results.get("raw_wacc"),
                    "relative_score": report_data.get("recommendation", {}).get("relative_score"),
                    "revenue_cagr": FCFFEngine(financial_df).calculate_revenue_cagr(),
                }
        except (KeyError, IndexError):
            whatif = None

    st.session_state["report"] = {
        "report_data": report_data,
        "ticker": context.ticker,
        "mode": context.mode,
        "peer_ticker": context.metadata.get("peer_ticker"),
        "pdf_bytes": context.pdf_bytes,
        "latency": end - start,
        "whatif": whatif,
    }

if "report" in st.session_state:
    _r = st.session_state["report"]
    report_data = _r["report_data"]
    recommendation = report_data.get("recommendation", {})
    rating = recommendation.get("rating", "Insufficient Data")
    color = RATING_COLORS.get(rating, "gray")

    if _r["mode"] == "comparison":
        st.info(f"Comparing {_r['ticker']} vs {_r['peer_ticker']}")

    st.markdown("## Report Ready")

    st.info(
        "**This report is a predictive analytical signal generated by an automated "
        "research tool, not personalized investment advice.** Past performance and "
        "model outputs are not guarantees of future results. Consult a licensed "
        "financial advisor before making investment decisions."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"### :{color}[{rating}]")
        st.caption(recommendation.get("basis", ""))
        if recommendation.get("composite_score") is not None:
            s1, s2, s3 = st.columns(3)
            s1.metric("DCF Score", f"{recommendation['dcf_score']:+.1f}")
            s2.metric(
                "Relative Score",
                f"{recommendation['relative_score']:+.1f}" if recommendation.get("relative_score") is not None else "N/A",
            )
            s3.metric("Composite Score", f"{recommendation['composite_score']:+.1f}")
        if recommendation.get("confidence_flag"):
            st.warning(f"**Low-confidence signal:** {recommendation['confidence_flag']}")
    with col2:
        confidence = report_data.get("confidence_scores", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall Score", confidence.get("Overall Score", "N/A"))
        c2.metric("Grounding", f"{confidence.get('Grounding (%)', 'N/A')}")
        c3.metric("Latency", f"{_r['latency']:.1f}s")

    narrative = report_data.get("narrative", {})
    if narrative.get("Executive Summary"):
        with st.expander("Executive Summary preview", expanded=True):
            st.write(narrative["Executive Summary"])

    monte_carlo = report_data.get("valuation_analysis", {}).get("monte_carlo")
    if monte_carlo:
        with st.expander("Monte Carlo Simulation (intrinsic value distribution)", expanded=False):
            st.caption(
                f"{monte_carlo['n_samples']:,} samples over growth rate / WACC / terminal growth -- "
                "a distribution around the DCF point estimate, not another verdict. Not yet folded into "
                "the recommendation composite above."
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Mean Intrinsic Value", f"${monte_carlo['mean']:,.2f}")
            m2.metric("Median", f"${monte_carlo['median']:,.2f}")
            m3.metric("Probability Undervalued", f"{monte_carlo['prob_undervalued']:.1%}")
            st.caption(
                f"90% CI: ${monte_carlo['ci_lower']:,.2f} - ${monte_carlo['ci_upper']:,.2f}  |  "
                f"25th-75th percentile: ${monte_carlo['p25']:,.2f} - ${monte_carlo['p75']:,.2f}  |  "
                f"Std Dev: ${monte_carlo['std_dev']:,.2f}"
            )

    ml_classifier = report_data.get("valuation_analysis", {}).get("ml_classifier")
    if ml_classifier:
        with st.expander("ML Valuation Classifier (informational only)", expanded=False):
            st.caption(
                "Trained on FinSight's own point-in-time backtest data. **Not part of the "
                "recommendation above** -- it has no accuracy track record yet, unlike DCF and "
                "relative valuation, both of which have been backtested."
            )
            top_prob = ml_classifier["probabilities"].get(ml_classifier["verdict"], 0)
            c1, c2 = st.columns(2)
            c1.metric("ML Verdict", ml_classifier["verdict"])
            c2.metric("Confidence", f"{top_prob:.1%}")
            st.caption(f"Model: {ml_classifier['model_name'].replace('_', ' ').title()}")

    whatif = _r.get("whatif")
    if whatif:
        with st.expander("What-If: Adjust DCF Assumptions", expanded=False):
            st.caption(
                "Drag a slider to see intrinsic value and the composite score/verdict update live. "
                "**This is a what-if exploration, not the official recommendation above** -- relative "
                "valuation stays fixed at its actual computed value; only the DCF assumptions change."
            )

            # st.slider's `format` is a plain printf-style format applied
            # to the raw value passed in -- it does NOT scale a fraction
            # into a percentage display. Sliders are built in percentage-
            # POINT units (e.g. 5.0 meaning 5%) and converted to fraction
            # (/100) only when calling compute_what_if, which expects the
            # same fractional units DCFEngine/FCFFEngine use everywhere
            # else (e.g. 0.05 for 5%).
            base_wacc_pct = (whatif["raw_wacc"] or 0.10) * 100
            growth_default_pct = min(max(whatif["revenue_cagr"] * 100, GROWTH_RATE_MIN * 100), GROWTH_RATE_MAX * 100)

            wg1, wg2, wg3 = st.columns(3)
            with wg1:
                whatif_growth_pct = st.slider(
                    "Revenue Growth Rate (near-term)",
                    min_value=GROWTH_RATE_MIN * 100, max_value=GROWTH_RATE_MAX * 100,
                    value=float(growth_default_pct),
                    step=0.5, format="%.1f%%",
                    key="whatif_growth",
                )
            with wg2:
                whatif_wacc_pct = st.slider(
                    "WACC",
                    min_value=base_wacc_pct - WACC_OFFSET * 100, max_value=base_wacc_pct + WACC_OFFSET * 100,
                    value=float(base_wacc_pct),
                    step=0.25, format="%.2f%%",
                    key="whatif_wacc",
                )
            with wg3:
                whatif_terminal_pct = st.slider(
                    "Terminal Growth Rate",
                    min_value=TERMINAL_GROWTH_MIN * 100, max_value=TERMINAL_GROWTH_MAX * 100,
                    value=DEFAULT_TERMINAL_GROWTH_RATE * 100,
                    step=0.25, format="%.2f%%",
                    key="whatif_terminal",
                )

            result = compute_what_if(
                financial_df=whatif["financial_df"],
                total_debt=whatif["total_debt"],
                cash=whatif["cash"],
                shares_outstanding=whatif["shares_outstanding"],
                current_price=whatif["current_price"],
                growth_rate=whatif_growth_pct / 100,
                wacc=whatif_wacc_pct / 100,
                terminal_growth_rate=whatif_terminal_pct / 100,
                relative_score=whatif["relative_score"],
            )

            if result is None:
                st.caption("What-if recompute unavailable for this company.")
            else:
                if result["wacc_floored"]:
                    st.caption(
                        f"Note: WACC floored to {result['wacc_used']*100:.2f}% at this slider position "
                        f"to avoid terminal-value instability (same floor the real DCF uses)."
                    )

                wr_color = RATING_COLORS.get(result["rating"], "gray")
                st.markdown(f"**What-if verdict:** :{wr_color}[{result['rating']}]  (actual report: :{color}[{rating}])")

                v1, v2, v3 = st.columns(3)
                v1.metric(
                    "What-If Intrinsic Value",
                    f"${result['intrinsic_value']:,.2f}",
                    delta=f"{result['upside_percent']:+.1f}% vs current price",
                )
                v2.metric(
                    "What-If Composite Score",
                    f"{result['composite_score']:+.1f}",
                    delta=(
                        f"{result['composite_score'] - recommendation['composite_score']:+.1f} vs actual"
                        if recommendation.get("composite_score") is not None else None
                    ),
                )
                v3.metric("What-If DCF Score", f"{result['dcf_score']:+.1f}")

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

    if _r["pdf_bytes"]:
        st.download_button(
            label="Download Full Research Report (PDF)",
            data=_r["pdf_bytes"],
            file_name=f"{_r['ticker']}_FinSight_Research_Report.pdf",
            mime="application/pdf",
        )

st.caption("Built using an LLM Planner + RAG over live SEC filings, ChromaDB, FinBERT and DCF valuation tools")
