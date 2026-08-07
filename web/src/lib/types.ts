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
  "Enterprise Value"?: string | number;
  "Equity Value"?: string | number;
  "Intrinsic Value (per share)"?: string | number;
  "Current Price"?: string | number;
  "Upside (%)"?: string | number;
  WACC?: string;
  "Raw WACC"?: string | number;
  "WACC Floor Note"?: string | null;
  "Terminal Growth Rate"?: string | number;
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

export type NarrativeSection =
  | "Executive Summary"
  | "Business Analysis"
  | "Market and Earnings Analysis"
  | "Risk Analysis"
  | "Investment Thesis";

export interface ReportData {
  currency?: string;
  currency_symbol?: string;
  filing_evidence_note?: string | null;
  recommendation?: RecommendationData;
  confidence_scores?: {
    "Overall Score"?: number | string;
    "Grounding (%)"?: number | string;
    "Retrieval (%)"?: number | string;
    "Citation Coverage (%)"?: number | string;
    "Completeness (%)"?: number | string;
  };
  narrative?: Partial<Record<NarrativeSection, string>>;
  market_earnings_snapshot?: MarketEarningsSnapshot;
  valuation_analysis?: ValuationAnalysis;
  financial_statement_analysis?: Record<string, number | string>;
  ratio_analysis?: Record<string, number | string>;
  growth_analysis?: Record<string, number | string>;
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

// POST /v1/research/{job_id}/model-compare's response shape -- an
// on-demand second opinion from 3 independent models re-interpreting
// the SAME already-computed evidence (not a re-run of the pipeline).
// rating is "Insufficient Data" (RatingBadge already supports this)
// when a model's response couldn't be parsed or the model was
// unreachable, same degrade-gracefully convention as everywhere else.
export interface ModelOpinion {
  label: string;
  model: string;
  rating: "Buy" | "Hold" | "Sell" | "Insufficient Data";
  confidence: number | null;
  reasoning: string;
}

export interface ModelConsensus {
  rating: "Buy" | "Hold" | "Sell" | "Insufficient Data";
  agree_count: number;
  total: number;
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

// GET /v1/research/recent's list-row shape -- deliberately NOT the
// full ResearchResult/JobResponse (fetching every past report's full
// nested result just to render a list row would be wasteful); mirrors
// db.list_recent_jobs()'s SELECT columns exactly.
export interface ReportSummary {
  job_id: string;
  ticker: string;
  company_name: string | null;
  rating: string;
  started_at: number;
}

// GET /v1/watchlist's item shape. price/change_pct are null when that
// ticker's live quote fetch failed (see main.py's per-ticker
// try/except -- one bad symbol must not fail the whole list). rating
// is null when the user has never researched this ticker.
export interface WatchlistItem {
  ticker: string;
  price: number | null;
  change_pct: number | null;
  rating: string | null;
  added_at: number;
  next_earnings_date: string | null;
  next_ex_dividend_date: string | null;
  last_dividend_amount: number | null;
  last_split: { date: string; ratio: number } | null;
}

// GET /v1/market/indices' item shape -- a fixed, curated list (see
// main.py's INDEX_LIST), not user-editable. price/change_pct are null
// when that index's quote fetch failed (same per-item isolation as
// WatchlistItem).
// GET /v1/companies/suggest's item shape -- powers the Watchlist
// add-ticker input's autocomplete dropdown.
export interface CompanySuggestion {
  ticker: string;
  name: string;
}

export interface IndexQuote {
  name: string;
  ticker: string;
  price: number | null;
  change_pct: number | null;
}

// GET /v1/portfolio's item shape. Self-reported quantity/avg_cost --
// never synced from a real brokerage. market_value/unrealized_pnl are
// null when the live quote fetch failed (same per-item isolation as
// WatchlistItem/IndexQuote); cost_basis is always knowable since it
// only depends on the user's own reported quantity/avg_cost.
export interface PortfolioHolding {
  ticker: string;
  quantity: number;
  avg_cost: number;
  price: number | null;
  change_pct: number | null;
  cost_basis: number;
  market_value: number | null;
  unrealized_pnl: number | null;
  unrealized_pnl_pct: number | null;
  added_at: number;
}

// GET /v1/news/market's item shape -- general market headlines, not
// tied to any one company (see app/reporting/news_client.py's
// fetch_market_news, distinct from the per-ticker company-news path).
export interface NewsArticle {
  headline: string;
  source: string;
  date: string;
  url: string;
  summary: string;
}

export interface PortfolioSummary {
  total_market_value: number | null;
  total_cost_basis: number | null;
  total_unrealized_pnl: number | null;
  total_unrealized_pnl_pct: number | null;
}
