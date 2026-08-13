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
  // Category -> {label: value}, from app/analysis/alpha_factors.py.
  // Deliberately loose on the value type: most entries are number |
  // string | null, but "P/E vs Own History"/"P/B vs Own History" are
  // small nested objects, and a category may carry an underscore-
  // prefixed "_..._note"/"_..._definition" string key -- see
  // ReportView.tsx's formatFieldValue/KeyValueGrid for how each shape
  // renders.
  alpha_factors?: Record<string, Record<string, unknown>> | null;
  [key: string]: unknown;
}

export interface InstitutionalConsensus {
  recommendation_consensus?: {
    score: number;
    label: string;
    methodology: string;
    institutional_ratings: { firm: string; rating: string; raw_grade?: string; date?: string }[];
    finsight_rating: string;
    summary: string;
    agreeing_count?: number;
    disagreeing_count?: number;
    total_count?: number;
    // See app/reporting/consensus_score.py's LOW_SAMPLE_THRESHOLD -- a
    // score from very few covering institutions is statistically less
    // meaningful than the same score from broad coverage.
    low_sample_size?: boolean;
    sample_size_note?: string | null;
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

// From app/reporting/report_data_builder.py's derive_signal_quality() --
// display-only consolidation of signals already computed elsewhere in
// the report (never feeds back into `recommendation`). confidence/
// evidence_coverage are 0-100 or null when there's nothing to be
// confident about; model_agreement is "N/M" or "N/A" when no votes
// were available (e.g. Insufficient Data rating).
export interface SignalQuality {
  quality_label: "HIGH" | "MEDIUM" | "LOW";
  confidence: number | null;
  evidence_coverage: number | null;
  model_agreement: string;
  breakdown: {
    valuation: "Bullish" | "Bearish" | "Neutral";
    fundamentals: "Bullish" | "Bearish" | "Neutral";
    momentum: "Bullish" | "Bearish" | "Neutral";
    sentiment: "Bullish" | "Bearish" | "Neutral";
    institutional_consensus: "Bullish" | "Bearish" | "Neutral";
  };
}

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
  signal_quality?: SignalQuality;
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
export interface PortfolioFit {
  sector: string | null;
  current_allocation_pct: number | null;
  summary: string;
}

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

// GET /v1/research/backtest-accuracy's response shape -- this app's
// own measured Buy/Hold/Sell accuracy against real historical
// outcomes (scripts/phase2_backtest.py's saved results), the same
// number for every report (not per-ticker -- see backtest_stats.py's
// own docstring for why). secondary is an older, out-of-sample cohort
// shown for honesty, not hidden just because it scored lower.
export interface BacktestAccuracySummary {
  accuracy_pct: number;
  correct: number;
  scored: number;
  window_label: string;
  secondary: {
    accuracy_pct: number;
    correct: number;
    scored: number;
    window_label: string;
  } | null;
}

// GET /v1/stocks/{ticker}/overview's response shape -- powers
// web/src/app/stock/[ticker]/page.tsx's header/price-statistics/
// fundamentals/company-info cards. Every numeric field here is null
// when yfinance's `.info` didn't have it for this ticker (a wiring gap
// fixed, not a guarantee every ticker populates every field) -- see
// app/data/stock_overview.py's own docstring for exactly which paid-
// vendor-only fields (shareholding pattern, promoter pledge, industry
// P/E, ...) are deliberately NOT here at all.
export interface StockOverview {
  ticker: string;
  is_crypto: boolean;
  price: number;
  change_pct: number | null;
  previous_close: number | null;
  currency: string;
  company_name: string | null;
  sector: string | null;
  industry: string | null;
  market_cap: number | null;
  business_summary: string | null;
  website: string | null;
  employees: number | null;
  country: string | null;
  exchange: string | null;
  next_earnings_date: string | null;
  price_statistics: {
    open: number | null;
    day_high: number | null;
    day_low: number | null;
    fifty_two_week_high: number | null;
    fifty_two_week_low: number | null;
    volume: number | null;
    average_volume: number | null;
  };
  fundamentals: {
    trailing_pe: number | null;
    forward_pe: number | null;
    price_to_book: number | null;
    dividend_yield: number | null;
    book_value: number | null;
    debt_to_equity: number | null;
    trailing_eps: number | null;
    peg_ratio: number | null;
  };
  analyst: {
    target_mean_price: number | null;
    target_high_price: number | null;
    target_low_price: number | null;
    number_of_analyst_opinions: number | null;
  };
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
  currency: string;
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
  region: "india" | "global";
  currency: string;
  price: number | null;
  change_pct: number | null;
}

// GET /v1/market/fx's response shape -- the USD/INR currency tracker.
// rate/change_pct are null when the live FX quote fetch failed.
export interface FxRate {
  pair: string;
  rate: number | null;
  change_pct: number | null;
}

// GET /v1/corporate-actions/feed's item shape -- one upcoming event
// (earnings or ex-dividend) for a ticker on the watchlist and/or
// portfolio. See app/api/main.py's get_corporate_actions_feed: "all"
// scope unions both ticker sources, "portfolio" is holdings only.
export interface CorporateActionEvent {
  ticker: string;
  name: string;
  type: "earnings" | "ex_dividend";
  date: string;
}

export interface CorporateActionsFeedData {
  events: CorporateActionEvent[];
  scope: "all" | "portfolio";
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
  buy_date: string | null;
  price: number | null;
  change_pct: number | null;
  // Native currency this holding actually trades in (from its live
  // quote) -- price/cost_basis/market_value/today_pnl below are ALL in
  // this currency, never converted. Only PortfolioSummary's totals are
  // USD-converted (see its own "currency"/"mixed_currency" fields).
  currency: string;
  cost_basis: number;
  market_value: number | null;
  unrealized_pnl: number | null;
  unrealized_pnl_pct: number | null;
  today_pnl: number | null;
  // Live verdict from this ticker's latest completed report -- the
  // SAME lookup WatchlistItem's rating uses, so a holding whose call
  // has since flipped (e.g. to Sell) shows it here too.
  rating: string | null;
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
  // Only present on GET /v1/news/my-stocks' articles -- which tracked
  // ticker this headline came from, so the "On My Stocks" tab can show it.
  ticker?: string;
  // Only present on GET /v1/stocks/{ticker}/news's articles -- keyword-
  // matched risk categories (see app/reporting/news_client.py), not an
  // LLM/ML sentiment score.
  categories?: string[];
}

// GET /v1/market/movers' item shape -- a row in the Top Movers
// "Tracked Universe" (curated backtest tickers + every user's
// watchlist tickers, unioned globally -- see
// app/reasoning/market_movers.py's module docstring). NOT full-market
// coverage; the frontend must always label this clearly.
export interface MoverItem {
  ticker: string;
  name: string;
  price: number;
  change_pct: number;
  currency: string;
}

export interface MarketMoversData {
  gainers: MoverItem[];
  losers: MoverItem[];
}

// GET /v1/market/sentiment's response shape -- FinSight's OWN research
// sentiment (Buy vs Sell share of every distinct ticker's latest
// completed rating across all users), not a market-wide indicator.
// buy_pct/sell_pct are null when nothing has been rated yet.
export interface MarketSentiment {
  buy_pct: number | null;
  sell_pct: number | null;
  buy_count: number;
  hold_count: number;
  sell_count: number;
  total_rated: number;
}

// GET /v1/portfolio/analysis's response shape -- a rollup of each
// holding's ALREADY-completed report rating (never triggers new
// research itself), both by simple count and by USD-equivalent
// market-value weight. value_weighted_pct is null until at least one
// holding both has a rating AND a valid market value to weight by.
export interface PortfolioAnalysis {
  holdings_total: number;
  holdings_researched: number;
  rating_counts: { Buy: number; Hold: number; Sell: number };
  value_weighted_pct: { Buy: number; Hold: number; Sell: number } | null;
  unresearched_tickers: string[];
}

// POST /v1/orders' response shape -- a simulated market order's
// instant fill. NO real broker, NO real money -- see
// db.execute_order's own docstring for the full boundary explanation.
// new_holding_quantity is the resulting position size after this
// order applied (0 if a SELL fully closed it out).
export interface OrderResult {
  order_id: string;
  ticker: string;
  side: "BUY" | "SELL";
  quantity: number;
  execution_price: number;
  currency: string;
  executed_at: number;
  new_holding_quantity: number;
}

// GET /v1/orders' item shape -- one row of order history.
export interface Order {
  order_id: string;
  ticker: string;
  side: "BUY" | "SELL";
  quantity: number;
  execution_price: number;
  currency: string;
  executed_at: number;
}

export interface PortfolioSummary {
  total_market_value: number | null;
  total_cost_basis: number | null;
  total_unrealized_pnl: number | null;
  total_unrealized_pnl_pct: number | null;
  total_today_pnl: number | null;
  total_today_pnl_pct: number | null;
  // Summary totals are always in this currency (USD), converted via a
  // live FX rate when a holding's own native currency differs --
  // mixed_currency flags that at least one holding needed conversion,
  // so the UI can label the total "(USD equiv.)" rather than implying
  // every holding actually trades in dollars. excluded_from_summary
  // flags a holding whose currency had no known FX rate (left out of
  // the totals entirely, still shown correctly on its own row).
  currency: string;
  mixed_currency: boolean;
  excluded_from_summary: boolean;
}
