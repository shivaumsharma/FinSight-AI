"use client";

import { useState } from "react";
import BacktestBadge from "./BacktestBadge";
import ModelCompare from "./ModelCompare";
import RatingBadge, { ratingColorClass } from "./RatingBadge";
import Tabs from "./Tabs";
import type { ResearchResult, SignalQuality } from "@/lib/types";

function ShareButton({ jobId }: { jobId: string }) {
  const [state, setState] = useState<"idle" | "sharing" | "copied" | "error">("idle");

  async function handleShare() {
    setState("sharing");
    try {
      const resp = await fetch(`/api/research/${jobId}/pdf/share`, { method: "POST" });
      if (!resp.ok) throw new Error("share request failed");
      const { url } = await resp.json();
      await navigator.clipboard.writeText(url);
      setState("copied");
      setTimeout(() => setState("idle"), 2000);
    } catch {
      setState("error");
      setTimeout(() => setState("idle"), 2000);
    }
  }

  const label = { idle: "SHARE", sharing: "...", copied: "LINK COPIED", error: "COULDN'T SHARE" }[state];

  return (
    <button
      type="button"
      onClick={handleShare}
      disabled={state === "sharing"}
      className="mt-6 ml-2 inline-block rounded-lg border border-border bg-card px-5 py-2.5 font-mono text-xs font-bold text-text hover:border-accent disabled:opacity-60"
    >
      {label}
    </button>
  );
}

function WatchlistButton({ ticker }: { ticker: string }) {
  const [state, setState] = useState<"idle" | "adding" | "added" | "error">("idle");

  async function handleAdd() {
    setState("adding");
    try {
      const resp = await fetch("/api/watchlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker }),
      });
      if (!resp.ok) throw new Error("watchlist add failed");
      setState("added");
    } catch {
      setState("error");
      setTimeout(() => setState("idle"), 2000);
    }
  }

  const label = { idle: "+ WATCHLIST", adding: "...", added: "ON WATCHLIST", error: "COULDN'T ADD" }[state];

  return (
    <button
      type="button"
      onClick={handleAdd}
      disabled={state === "adding" || state === "added"}
      className="mt-6 ml-2 inline-block rounded-lg border border-border bg-card px-5 py-2.5 font-mono text-xs font-bold text-text hover:border-accent disabled:opacity-60"
    >
      {label}
    </button>
  );
}

function StatTile({ label, value, valueClass }: { label: string; value: React.ReactNode; valueClass?: string }) {
  return (
    <div className="flex-1 rounded-lg border border-border bg-card px-3 py-2.5">
      <div className="font-mono text-[10px] text-muted">{label}</div>
      <div className={`mt-0.5 font-mono text-sm font-bold ${valueClass || "text-text"}`}>{value}</div>
    </div>
  );
}

// Separates *direction* (the rating above, already final) from *quality
// of evidence* -- see app/reporting/report_data_builder.py's
// derive_signal_quality() docstring. Display-only: never changes the
// rating, same visual pattern (label + StatTile row + a small grid) as
// FactorCategory's Alpha Factors blocks below.
function SignalQualityBlock({ signalQuality }: { signalQuality: SignalQuality | undefined }) {
  if (!signalQuality) return null;

  const labelClass =
    signalQuality.quality_label === "HIGH"
      ? "text-accent"
      : signalQuality.quality_label === "LOW"
      ? "text-danger"
      : "text-warn";

  const directionClass = (direction: string) =>
    direction === "Bullish" ? "text-accent" : direction === "Bearish" ? "text-danger" : "text-warn";

  return (
    <div className="mt-3 rounded-lg border border-border bg-card px-3.5 py-3">
      <div className="flex items-center justify-between">
        <div className="font-mono text-[11px] font-bold tracking-wide text-muted">SIGNAL QUALITY</div>
        <div className={`font-mono text-xs font-bold ${labelClass}`}>{signalQuality.quality_label}</div>
      </div>
      <div className="mt-2 flex gap-2">
        <StatTile
          label="CONFIDENCE"
          value={signalQuality.confidence !== null ? `${signalQuality.confidence}%` : "N/A"}
        />
        <StatTile
          label="EVIDENCE COVERAGE"
          value={signalQuality.evidence_coverage !== null ? `${signalQuality.evidence_coverage}%` : "N/A"}
        />
        <StatTile label="MODEL AGREEMENT" value={signalQuality.model_agreement} />
      </div>
      <div className="mt-2.5 grid grid-cols-2 gap-x-4 gap-y-1.5 sm:grid-cols-3">
        {Object.entries(signalQuality.breakdown).map(([key, direction]) => (
          <div key={key} className="flex items-center justify-between text-[11px]">
            <span className="text-muted">{key.replace(/_/g, " ").toUpperCase()}</span>
            <span className={`font-mono font-bold ${directionClass(direction)}`}>{direction}</span>
          </div>
        ))}
      </div>
      <p className="mt-2.5 text-[10px] text-dim">
        How much of the supporting evidence and independent cross-checks agree -- separate from, and never a
        factor in, the rating above.
      </p>
    </div>
  );
}

function ToneTile({ label, tone }: { label: string; tone: string }) {
  const lower = (tone || "").toLowerCase();
  const cls = lower.includes("positive")
    ? "text-accent"
    : lower.includes("negative")
    ? "text-danger"
    : "text-warn";
  return (
    <div className="flex flex-1 items-center justify-between rounded-lg border border-border bg-card px-3 py-2">
      <span className="font-mono text-[10px] text-muted">{label}</span>
      <span className={`font-mono text-[10px] font-bold uppercase ${cls}`}>{tone || "Unavailable"}</span>
    </div>
  );
}

// report_data_builder.py hands back raw numbers for every field here --
// nothing is pre-formatted server-side (confirmed against a real job
// response: WACC=0.1100821..., "Intrinsic Value (per share)"=96.6947...).
// The old Streamlit UI never rendered these fields directly (only the
// PDF builder formatted them), so this formatting never existed until
// now. Dispatches on the field's *name*, not a value heuristic, since
// the key vocabulary is fixed and small (report_data_builder.py's
// build_report_data()) -- a fraction like WACC and a percentage-point
// value like "Revenue Growth (%)" are both plain floats and
// indistinguishable by magnitude alone.
const FRACTION_KEYS = new Set(["WACC", "Raw WACC", "Terminal Growth Rate"]);
const DOLLAR_LARGE_KEYS = new Set(["Revenue", "EBIT", "Net Income", "Free Cash Flow", "Enterprise Value", "Equity Value"]);
const DOLLAR_SMALL_KEYS = new Set(["Current Price", "Intrinsic Value (per share)", "EPS"]);

function fmtLargeDollar(v: number, symbol: string): string {
  const abs = Math.abs(v);
  if (abs >= 1e9) return `${symbol}${(v / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${symbol}${(v / 1e6).toFixed(2)}M`;
  return `${symbol}${v.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
}

// "P/E vs Own History"/"P/B vs Own History" (app/analysis/alpha_factors.py)
// are the one case a field value is a small nested object rather than a
// plain number/string -- same shape pdf_report_builder.py's own
// _alpha_factor_value renders, kept in sync with that compact form.
function isVsHistoryValue(value: unknown): value is {
  current?: number; historical_avg?: number; years_used?: number; vs_history_pct?: number; signal?: string;
} {
  return (
    typeof value === "object" &&
    value !== null &&
    "vs_history_pct" in value &&
    "signal" in value
  );
}

function formatFieldValue(key: string, value: unknown, symbol: string): string {
  if (value === null || value === undefined) return "N/A";
  if (isVsHistoryValue(value)) {
    const pct = value.vs_history_pct ?? 0;
    return `${value.current}x vs ${value.historical_avg}x ${value.years_used}yr avg (${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%, ${value.signal})`;
  }
  if (typeof value !== "number") return String(value);

  if (FRACTION_KEYS.has(key)) return `${(value * 100).toFixed(2)}%`;
  if (key.endsWith("(%)")) return `${value.toFixed(2)}%`;
  if (DOLLAR_LARGE_KEYS.has(key)) return fmtLargeDollar(value, symbol);
  if (DOLLAR_SMALL_KEYS.has(key)) return `${symbol}${value.toFixed(2)}`;
  if (key === "Debt to Equity") return `${value.toFixed(2)}x`;
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

// Underscore-prefixed keys (e.g. alpha_factors' "_market_note"/
// "_macro_note"/"_interest_rate_sensitivity_definition") are captions,
// not fields -- excluded here and rendered separately by FactorCategory
// below. No existing caller's data has ever used a leading underscore
// key, so this filter is a no-op everywhere else.
function KeyValueGrid({ data, symbol }: { data: Record<string, unknown> | undefined; symbol: string }) {
  if (!data) return null;
  const entries = Object.entries(data).filter(([k, v]) => !k.startsWith("_") && v !== null && v !== undefined);
  return (
    <div className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3">
      {entries.map(([k, v]) => (
        <div key={k}>
          <div className="font-mono text-[10px] uppercase tracking-wide text-muted">{k}</div>
          <div className="font-mono text-sm text-text">{formatFieldValue(k, v, symbol)}</div>
        </div>
      ))}
    </div>
  );
}

// One Alpha Factors category block (Financial/Quality/Valuation/Market/
// Risk/Sentiment/Macro) -- label + KeyValueGrid + an optional note
// caption pulled out of that category's own underscore-prefixed key.
// Renders nothing for a category with zero real fields (e.g. Macro for
// an .NS ticker, where Interest Rate Sensitivity is gated to null and
// only the note remains) -- a lone caption with no data above it would
// read as broken, not informative.
function FactorCategory({ label, data, symbol }: { label: string; data: Record<string, unknown> | undefined; symbol: string }) {
  if (!data) return null;
  const hasFields = Object.entries(data).some(([k, v]) => !k.startsWith("_") && v !== null && v !== undefined);
  if (!hasFields) return null;
  const note = Object.entries(data).find(([k, v]) => k.startsWith("_") && typeof v === "string")?.[1] as
    | string
    | undefined;
  return (
    <div>
      <div className="mb-2 font-mono text-[11px] font-bold tracking-wide text-muted">{label.toUpperCase()}</div>
      <KeyValueGrid data={data} symbol={symbol} />
      {note && <p className="mt-2 text-[10px] leading-relaxed text-dim">{note}</p>}
    </div>
  );
}

function fmtScore(n: number | null | undefined): string {
  if (n === null || n === undefined) return "N/A";
  return `${n >= 0 ? "+" : ""}${n.toFixed(1)}`;
}

function fmtMoney(n: number | null | undefined, symbol: string): string {
  if (n === null || n === undefined) return "N/A";
  return `${symbol}${n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

const NARRATIVE_SECTIONS = [
  "Executive Summary",
  "Business Analysis",
  "Market and Earnings Analysis",
  "Risk Analysis",
  "Investment Thesis",
] as const;

export default function ReportView({
  result,
  jobId,
  latencySeconds,
}: {
  result: ResearchResult;
  jobId: string;
  latencySeconds: number | null;
}) {
  const rd = result.report_data || {};
  const symbol = rd.currency_symbol || "$";
  const rec = rd.recommendation || { rating: "Insufficient Data", basis: "" };
  const confidence = rd.confidence_scores || {};
  const market = rd.market_earnings_snapshot || {};
  const valuation = rd.valuation_analysis || {};
  const monteCarlo = valuation.monte_carlo;
  const mlClassifier = valuation.ml_classifier;
  const alphaFactors = valuation.alpha_factors;
  const consensus = rd.institutional_consensus?.recommendation_consensus;
  const news = rd.news_sources || {};
  const company = (rd.company_overview as Record<string, unknown>) || {};

  const upside = valuation["Upside (%)"];
  const upsideNum = typeof upside === "string" ? parseFloat(upside) : (upside as number | undefined);

  return (
    <div className="mt-8">
      <div className="flex items-center justify-between">
        <div>
          <div className="font-mono text-lg font-bold text-text">{result.ticker}</div>
          <div className="text-xs text-muted">
            {String(company.name || "")}
            {company.sector ? ` · ${company.sector}` : ""}
          </div>
        </div>
        {result.mode === "comparison" && result.peer_ticker && (
          <div className="rounded bg-card px-3 py-1 text-xs text-muted">vs {result.peer_ticker}</div>
        )}
      </div>

      <div className="mt-4 rounded-lg border border-amber-900/60 bg-amber-950/40 px-4 py-3 text-xs text-warn">
        <strong>Predictive analytical signal, not personalized investment advice.</strong> Past performance and
        model outputs are not guarantees of future results. Consult a licensed financial advisor before making
        investment decisions.
      </div>

      {/* Verdict card -- border color set inline since it's chosen from
          a runtime value (rating); a Tailwind class built via template
          string wouldn't survive Tailwind's static build-time scan. */}
      <div
        className="mt-4 flex items-center justify-between rounded-lg border bg-card px-4 py-3"
        style={{
          borderColor:
            rec.rating === "Buy"
              ? "var(--accent)"
              : rec.rating === "Sell"
              ? "var(--danger)"
              : rec.rating === "Hold"
              ? "var(--warn)"
              : "var(--border)",
        }}
      >
        <div>
          <div className="flex items-center gap-2">
            <RatingBadge rating={rec.rating} />
            <BacktestBadge />
          </div>
          <div className="mt-1 text-[10.5px] text-muted">
            composite {fmtScore(rec.composite_score)} · overall score {confidence["Overall Score"] ?? "N/A"}
          </div>
        </div>
        {upsideNum !== undefined && !Number.isNaN(upsideNum) && (
          <div className="text-right">
            <div className="font-mono text-[10px] text-muted">UPSIDE</div>
            <div className={`font-mono text-lg font-bold ${upsideNum >= 0 ? "text-accent" : "text-danger"}`}>
              {upsideNum >= 0 ? "+" : ""}
              {upsideNum.toFixed(1)}%
            </div>
          </div>
        )}
      </div>
      <p className="mt-2 text-xs text-muted">{rec.basis}</p>
      {rec.confidence_flag && (
        <div className="mt-2 rounded-lg border border-amber-900/60 bg-amber-950/40 px-3 py-2 text-xs text-warn">
          <strong>Low-confidence signal:</strong> {rec.confidence_flag}
        </div>
      )}
      <SignalQualityBlock signalQuality={rd.signal_quality} />

      {/* Stat tiles */}
      <div className="mt-3 flex gap-2">
        <StatTile label="PRICE" value={fmtMoney(market.current_price, symbol)} />
        <StatTile label="INTRINSIC" value={formatFieldValue("Intrinsic Value (per share)", valuation["Intrinsic Value (per share)"], symbol)} />
        <StatTile label="WACC" value={formatFieldValue("WACC", valuation["WACC"], symbol)} />
      </div>
      <div className="mt-2 flex gap-2">
        <ToneTile label="SEC TONE (MGMT)" tone={market.sentiment_label || ""} />
        <ToneTile label="NEWS TONE" tone={market.news_sentiment_label || ""} />
      </div>
      {rd.filing_evidence_note && (
        <div className="mt-2 rounded-lg border border-amber-900/60 bg-amber-950/40 px-3 py-2 text-xs text-warn">
          <strong>Note:</strong> {rd.filing_evidence_note}
        </div>
      )}
      {latencySeconds !== null && (
        <div className="mt-2 text-right text-[10px] font-mono text-dim">generated in {latencySeconds.toFixed(1)}s</div>
      )}

      {/* Tabs */}
      <div className="mt-6">
        <Tabs
          tabs={[
            {
              label: "Overview",
              content: (
                <div>
                  <div className="mb-2 font-mono text-[11px] font-bold tracking-wide text-muted">
                    EXECUTIVE SUMMARY
                  </div>
                  <p className="whitespace-pre-wrap text-sm leading-relaxed text-text/90">
                    {rd.narrative?.["Executive Summary"] || "Not available."}
                  </p>
                </div>
              ),
            },
            {
              label: "Financials",
              content: (
                <div className="space-y-5">
                  <div>
                    <div className="mb-2 font-mono text-[11px] font-bold tracking-wide text-muted">STATEMENT</div>
                    <KeyValueGrid data={rd.financial_statement_analysis} symbol={symbol} />
                  </div>
                  <div>
                    <div className="mb-2 font-mono text-[11px] font-bold tracking-wide text-muted">RATIOS</div>
                    <KeyValueGrid data={rd.ratio_analysis} symbol={symbol} />
                  </div>
                  <div>
                    <div className="mb-2 font-mono text-[11px] font-bold tracking-wide text-muted">GROWTH</div>
                    <KeyValueGrid data={rd.growth_analysis} symbol={symbol} />
                  </div>
                </div>
              ),
            },
            {
              label: "DCF",
              content: (
                <div className="space-y-5">
                  {valuation["DCF Available"] === false ? (
                    <p className="text-sm text-muted">{String(valuation["DCF Unavailable Reason"] || "DCF unavailable for this company.")}</p>
                  ) : (
                    <KeyValueGrid
                      data={{
                        "Enterprise Value": valuation["Enterprise Value"],
                        "Equity Value": valuation["Equity Value"],
                        "Intrinsic Value (per share)": valuation["Intrinsic Value (per share)"],
                        "Current Price": valuation["Current Price"],
                        "Upside (%)": valuation["Upside (%)"],
                        WACC: valuation["WACC"],
                        "Raw WACC": valuation["Raw WACC"],
                        "Terminal Growth Rate": valuation["Terminal Growth Rate"],
                      }}
                      symbol={symbol}
                    />
                  )}

                  {monteCarlo && (
                    <div>
                      <div className="mb-2 font-mono text-[11px] font-bold tracking-wide text-muted">
                        MONTE CARLO ({monteCarlo.n_samples.toLocaleString()} samples)
                      </div>
                      <div className="flex gap-2">
                        <StatTile label="MEAN" value={fmtMoney(monteCarlo.mean, symbol)} />
                        <StatTile label="MEDIAN" value={fmtMoney(monteCarlo.median, symbol)} />
                        <StatTile label="P(UNDERVALUED)" value={`${(monteCarlo.prob_undervalued * 100).toFixed(1)}%`} />
                      </div>
                      <p className="mt-2 text-[11px] text-muted">
                        90% CI: {fmtMoney(monteCarlo.ci_lower, symbol)} - {fmtMoney(monteCarlo.ci_upper, symbol)}
                      </p>
                    </div>
                  )}

                  {mlClassifier && (
                    <div>
                      <div className="mb-2 font-mono text-[11px] font-bold tracking-wide text-muted">
                        ML CLASSIFIER (informational only)
                      </div>
                      <div className="flex gap-2">
                        <StatTile label="VERDICT" value={mlClassifier.verdict} />
                        <StatTile
                          label="CONFIDENCE"
                          value={`${((mlClassifier.probabilities[mlClassifier.verdict] || 0) * 100).toFixed(1)}%`}
                        />
                      </div>
                    </div>
                  )}
                </div>
              ),
            },
            {
              label: "Factors",
              content: alphaFactors ? (
                <div className="space-y-5">
                  <p className="text-[11px] leading-relaxed text-muted">
                    A transparent scorecard of financial, valuation, quality, market, risk, sentiment, and
                    macro signals shown for context alongside the DCF -- informational only, not part of the
                    recommendation above.
                  </p>
                  <FactorCategory label="Financial" data={alphaFactors.financial} symbol={symbol} />
                  <FactorCategory label="Quality" data={alphaFactors.quality} symbol={symbol} />
                  <FactorCategory label="Valuation" data={alphaFactors.valuation} symbol={symbol} />
                  <FactorCategory label="Market" data={alphaFactors.market} symbol={symbol} />
                  <FactorCategory label="Risk" data={alphaFactors.risk} symbol={symbol} />
                  <FactorCategory label="Sentiment" data={alphaFactors.sentiment} symbol={symbol} />
                  <FactorCategory label="Macro" data={alphaFactors.macro} symbol={symbol} />
                </div>
              ) : (
                <p className="text-sm text-muted">Not available for this report.</p>
              ),
            },
            {
              label: "Narrative",
              content: (
                <div className="space-y-5">
                  {NARRATIVE_SECTIONS.map((section) =>
                    rd.narrative?.[section] ? (
                      <div key={section}>
                        <div className="mb-2 font-mono text-[11px] font-bold tracking-wide text-muted">
                          {section.toUpperCase()}
                        </div>
                        <p className="whitespace-pre-wrap text-sm leading-relaxed text-text/90">{rd.narrative[section]}</p>
                      </div>
                    ) : null
                  )}
                </div>
              ),
            },
            {
              label: "Sources",
              content: (
                <div className="space-y-6">
                  {consensus && (
                    <div>
                      <div className="mb-2 font-mono text-[11px] font-bold tracking-wide text-muted">
                        INSTITUTIONAL CONSENSUS
                      </div>
                      <div className="font-mono text-2xl font-bold text-text">{consensus.score}%</div>
                      <div className="text-xs text-muted">{consensus.label}</div>
                      <p className="mt-2 text-[11px] text-muted">{consensus.methodology}</p>
                      {consensus.low_sample_size && consensus.sample_size_note && (
                        <div className="mt-2 rounded-lg border border-amber-900/60 bg-amber-950/40 px-3 py-2 text-[11px] text-warn">
                          <strong>Note:</strong> {consensus.sample_size_note}
                        </div>
                      )}
                      <div className="mt-3 space-y-1">
                        {consensus.institutional_ratings.map((r, i) => (
                          <div key={i} className="flex items-baseline justify-between text-xs">
                            <span className="text-text/80">
                              {r.firm}
                              {r.date && <span className="ml-1.5 text-[10px] text-dim">{r.date}</span>}
                            </span>
                            <span className="flex items-baseline gap-1.5">
                              {r.raw_grade && <span className="text-[10px] text-muted">{r.raw_grade}</span>}
                              <span
                                className={`font-mono font-bold ${ratingColorClass(
                                  r.rating.charAt(0).toUpperCase() + r.rating.slice(1).toLowerCase()
                                )}`}
                              >
                                {r.rating}
                              </span>
                            </span>
                          </div>
                        ))}
                      </div>
                      <p className="mt-3 text-xs text-muted">{consensus.summary}</p>
                    </div>
                  )}

                  <div>
                    <div className="mb-2 font-mono text-[11px] font-bold tracking-wide text-muted">
                      NEWS SOURCES ({news.total_retrieved || 0} retrieved, {news.total_selected || 0} used)
                    </div>
                    {!news.total_retrieved ? (
                      <p className="text-xs text-muted">No recent news coverage was found for this company.</p>
                    ) : (
                      <div className="space-y-2">
                        {news.all_articles?.map((a, i) => (
                          <div key={i} className="text-xs">
                            <a href={a.url} target="_blank" rel="noreferrer" className="font-medium text-accent hover:underline">
                              {a.headline}
                            </a>
                            <div className="text-[10px] text-muted">
                              {a.source} — {a.date} — {a.used_in_analysis ? "Used" : "Retrieved, not used"}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ),
            },
          ]}
        />
      </div>

      <a
        href={`/api/research/${jobId}/pdf`}
        download={`${result.ticker}_FinSight_Research_Report.pdf`}
        className="mt-6 inline-block rounded-lg bg-accent px-5 py-2.5 font-mono text-xs font-bold text-bg hover:opacity-90"
      >
        EXPORT PDF
      </a>
      <ShareButton jobId={jobId} />
      <WatchlistButton ticker={result.ticker} />
      <ModelCompare jobId={jobId} />
    </div>
  );
}
