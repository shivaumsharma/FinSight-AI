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

    transcript_path: Optional[str] = None
    transcript_text: Optional[str] = None
    transcript_chunks: List[str] = field(default_factory=list)
    retrieved_chunks: List[str] = field(default_factory=list)
    citations: List[Dict[str, str]] = field(default_factory=list)

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