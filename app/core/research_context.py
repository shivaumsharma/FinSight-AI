"""
research_context.py

Central data object passed through the Financial Research Pipeline.

Flow:

User Question
      │
      ▼
ResearchContext
      │
      ├── Market Data
      ├── Company Information
      ├── Financial Statements
      ├── RAG Evidence
      ├── Sentiment
      ├── Valuation
      ├── Generated Analysis
      └── Evaluation Metrics
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
from datetime import datetime


@dataclass
class ResearchContext:
    """
    Shared context object used across the entire research pipeline.
    Every module should read from and write to this object instead
    of passing dozens of variables around.
    """

    # ==========================================================
    # Request Information
    # ==========================================================

    ticker: str
    question: str

    request_time: datetime = field(default_factory=datetime.utcnow)

    # ==========================================================
    # Company Information
    # ==========================================================

    company_info: Dict[str, Any] = field(default_factory=dict)

    # ==========================================================
    # Financial Data
    # ==========================================================

    income_statement: Optional[pd.DataFrame] = None
    balance_sheet: Optional[pd.DataFrame] = None
    cash_flow: Optional[pd.DataFrame] = None

    normalized_financials: Optional[pd.DataFrame] = None

    historical_prices: Optional[pd.DataFrame] = None

    market_cap: Optional[float] = None
    beta: Optional[float] = None 
    
    # Raw Data

    financial_metrics: Dict[str, Any] = field(default_factory=dict)
    valuation_metrics: Dict[str, Any] = field(default_factory=dict)
    sentiment_metrics: Dict[str, Any] = field(default_factory=dict)

    # Human Readable Summaries

    financial_summary: str = ""
    valuation_summary: str = ""
    sentiment_summary: str = ""
    research_summary: str = ""
    # ==========================================================
    # RAG Pipeline
    # ==========================================================

    transcript_chunks: List[str] = field(default_factory=list)
    retrieved_chunks: List[str] = field(default_factory=list)
    citations: List[Dict[str, str]] = field(default_factory=list)

    # ==========================================================
    # News (see app/reporting/news_client.py, news_sentiment.py)
    # ==========================================================

    # Every article retrieved, unfiltered -- shown in full in the
    # "News Sources Used" transparency panel, not just the ones fed
    # to the LLM or explicitly cited in prose.
    news_articles: List[Dict[str, Any]] = field(default_factory=list)

    # Capped, category-diverse subset actually fed to the narrative
    # prompt (see news_client.select_for_analysis).
    news_selected: List[Dict[str, Any]] = field(default_factory=list)

    news_sentiment: Optional[Dict[str, Any]] = None
    news_sentiment_summary: Dict[str, Any] = field(default_factory=dict)

    # ==========================================================
    # NLP
    # ==========================================================

    sentiment: Dict[str, Any] = field(default_factory=dict)

    # ==========================================================
    # Valuation
    # ==========================================================

    valuation_results: Dict[str, Any] = field(default_factory=dict)

    enterprise_value: Optional[float] = None

    equity_value: Optional[float] = None

    intrinsic_value: Optional[float] = None

    # ==========================================================
    # LLM Output
    # ==========================================================

    generated_answer: Optional[str] = None

    investment_thesis: Dict[str, Any] = field(default_factory=dict)

    # Full structured report: deterministic sections (company overview,
    # ratios, growth, valuation, confidence scores, references) plus
    # narrative["<section name>"] for the five LLM-written sections --
    # see app/reporting/report_data_builder.py and narrative_builder.py.
    report_data: Optional[Dict[str, Any]] = None

    # Rendered PDF bytes, built from report_data -- see
    # app/reporting/pdf_report_builder.py.
    pdf_bytes: Optional[bytes] = None

    # Institutional Consensus Score: how closely FinSight's own
    # recommendation agrees with real institutional analyst ratings.
    # Evaluation/market-context only -- see
    # app/reporting/consensus_score.py's module docstring. None if no
    # institutional coverage exists for this ticker.
    institutional_consensus: Optional[Dict[str, Any]] = None

    # ==========================================================
    # Evaluation
    # ==========================================================

    evaluation: Dict[str, Any] = field(default_factory=dict)

    # Example:
    #
    # {
    #     "precision": 0.92,
    #     "recall": 0.88,
    #     "faithfulness": 0.95,
    #     "latency": 1.42
    # }

    # ==========================================================
    # Agentic Execution (new)
    # ==========================================================

    # Ordered record of which tools actually ran against this
    # context, populated by the ResearchAgent. Used for
    # observability/debugging and shown in the UI trace panel.
    tool_trace: List[str] = field(default_factory=list)

    # "single" (one company) or "comparison" (two companies).
    # Set by the ResearchAgent before tool execution begins.
    mode: str = "single"

    # ==========================================================
    # Metadata
    # ==========================================================

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ==========================================================
    # Helper Methods
    # ==========================================================

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def add_evaluation_metric(self, metric: str, value: Any) -> None:
        self.evaluation[metric] = value

    def record_tool(self, tool_name: str) -> None:
        """Append a tool name to the execution trace (idempotent per call)."""
        self.tool_trace.append(tool_name)

    def to_dict(self) -> Dict[str, Any]:
        """
        Lightweight serialization for logging.
        DataFrames remain unchanged and can be handled separately.
        """

        return {
            "ticker": self.ticker,
            "question": self.question,
            "company_info": self.company_info,
            "market_cap": self.market_cap,
            "beta": self.beta,
            "valuation_results": self.valuation_results,
            "generated_answer": self.generated_answer,
            "investment_thesis": self.investment_thesis,
            "evaluation": self.evaluation,
            "metadata": self.metadata,
            "request_time": self.request_time.isoformat(),
        }