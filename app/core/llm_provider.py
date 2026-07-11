"""
llm_provider.py

Process-wide singleton accessor for the local text-generation model
used both for report writing (ReportTool) and for planning
(LLMPlanner).

Why this exists
----------------
ReportGenerator now lazy-loads its HF pipeline (see
app/rag/report_generator.py). Without a shared accessor, ReportTool
and LLMPlanner would each construct their own ReportGenerator and
end up loading two independent copies of the same ~1.5B parameter
model into memory/GPU the first time each ran -- an easy and costly
mistake in an agentic system with several LLM-touching components.
`get_shared_generator()` guarantees exactly one instance per process.
"""

from app.rag.report_generator import ReportGenerator

_shared_generator = None


def get_shared_generator() -> ReportGenerator:
    global _shared_generator
    if _shared_generator is None:
        _shared_generator = ReportGenerator()
    return _shared_generator
