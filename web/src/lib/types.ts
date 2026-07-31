// Mirrors app/api/serialization.py's context_to_api_dict() output and
// app/api/errors.py's error codes. Deliberately loose on deeply nested,
// display-only sections (news_sources, institutional_consensus, etc.) --
// this is a rendering client, not a second source of truth for the
// pipeline's data shape, so over-typing those risks silently breaking
// on a backend field addition/rename that doesn't actually matter here.

export interface RecommendationData {
  rating: "Buy" | "Hold" | "Sell" | "Insufficient Data";
  basis: string;
  dcf_score?: number | null;
  relative_score?: number | null;
  composite_score?: number | null;
  confidence_flag?: string | null;
}

export interface MarketEarningsSnapshot {
  current_price?: number | null;
  market_cap?: number | null;
  sentiment_label?: string;
  sentiment_confidence?: string | number;
  news_sentiment_label?: string;
  news_sentiment_confidence?: string | number;
  next_earnings_date?: string | null;
}

export interface ValuationAnalysis {
  "DCF Available"?: boolean;
  "DCF Unavailable Reason"?: string | null;
  monte_carlo?: {
    n_samples: number;
    mean: number;
    median: number;
    prob_undervalued: number;
    ci_lower: number;
    ci_upper: number;
    p25: number;
    p75: number;
    std_dev: number;
  } | null;
  ml_classifier?: {
    verdict: string;
    probabilities: Record<string, number>;
    model_name: string;
  } | null;
  [key: string]: unknown;
}

export interface InstitutionalConsensus {
  recommendation_consensus?: {
    score: number;
    label: string;
    methodology: string;
    institutional_ratings: { firm: string; rating: string }[];
    finsight_rating: string;
    summary: string;
  } | null;
}

export interface NewsSources {
  total_retrieved?: number;
  total_selected?: number;
  all_articles?: {
    headline: string;
    source: string;
    date: string;
    url: string;
    used_in_analysis: boolean;
  }[];
}

export interface ReportData {
  recommendation?: RecommendationData;
  confidence_scores?: { "Overall Score"?: number | string; "Grounding (%)"?: number | string };
  narrative?: { "Executive Summary"?: string; [key: string]: unknown };
  market_earnings_snapshot?: MarketEarningsSnapshot;
  valuation_analysis?: ValuationAnalysis;
  institutional_consensus?: InstitutionalConsensus;
  news_sources?: NewsSources;
  [key: string]: unknown;
}

export interface ResearchResult {
  ticker: string;
  mode: string;
  peer_ticker?: string | null;
  report_data: ReportData;
  normalized_financials?: string | null;
  raw_wacc?: number | null;
  tool_trace?: string[];
  llm_usage?: Record<string, number> | null;
}

export type JobStatus = "queued" | "running" | "done" | "error";

export interface JobResponse {
  job_id: string;
  status: JobStatus;
  question: string;
  orchestrator: string;
  result?: ResearchResult;
  error_code?: string;
  error_message?: string;
}

export interface ApiErrorBody {
  code: string;
  message: string;
}
