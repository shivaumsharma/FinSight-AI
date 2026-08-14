"use client";

import { useEffect, useState } from "react";
import StatGrid from "./StatGrid";
import { currencySymbol } from "@/lib/currency";
import { fmtPrice } from "@/lib/stockFormat";

interface TechnicalsData {
  indicators: {
    rsi_14: number | null;
    macd: { macd: number | null; signal: number | null; histogram: number | null };
    adx_14: number | null;
    vwap: number | null;
    ema_20: number | null;
    ema_50: number | null;
    ema_200: number | null;
    stochastic: { k: number | null; d: number | null };
    williams_r: number | null;
  };
  trend: "Bullish" | "Bearish" | "Neutral";
  moving_average_signal: { buy: number; sell: number; verdict: "Buy" | "Sell" | "Neutral" };
  support_resistance: Record<string, number | null>;
}

function fmtNum(v: number | null | undefined, decimals = 2): string {
  if (v === null || v === undefined) return "N/A";
  return v.toFixed(decimals);
}

function TrendBadge({ trend }: { trend: TechnicalsData["trend"] }) {
  const cls = trend === "Bullish" ? "text-accent border-accent" : trend === "Bearish" ? "text-danger border-danger" : "text-muted border-dim";
  return <span className={`inline-block rounded border px-3 py-1 font-mono text-sm font-bold uppercase tracking-wide ${cls}`}>{trend}</span>;
}

function VerdictBadge({ verdict }: { verdict: "Buy" | "Sell" | "Neutral" }) {
  const cls = verdict === "Buy" ? "text-accent border-accent" : verdict === "Sell" ? "text-danger border-danger" : "text-muted border-dim";
  return <span className={`inline-block rounded border px-2 py-0.5 font-mono text-xs font-bold uppercase ${cls}`}>{verdict}</span>;
}

// Self-fetches /technicals independently from PriceChart -- both live
// on the same Technicals tab but PriceChart's own range selector would
// otherwise force this panel to re-fetch on every range click for no
// reason (indicator readouts don't need to track the chart's zoom
// window, just the default 1y).
export default function TechnicalsPanel({ ticker, currency }: { ticker: string; currency: string }) {
  const [data, setData] = useState<TechnicalsData | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    setData(null);
    setError(false);
    fetch(`/api/stock/${encodeURIComponent(ticker)}/technicals`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setData)
      .catch(() => setError(true));
  }, [ticker]);

  if (error) {
    return <p className="mt-4 py-6 text-center font-mono text-[11px] text-dim">Couldn&apos;t load technical data for this ticker.</p>;
  }
  if (!data) {
    return (
      <div className="mt-4 flex flex-col gap-2">
        <div className="h-[60px] animate-pulse rounded-lg bg-card/60" />
        <div className="h-[140px] animate-pulse rounded-lg bg-card/60" />
      </div>
    );
  }

  const ind = data.indicators;
  const sr = data.support_resistance;
  const symbol = currencySymbol(currency);

  return (
    <div className="mt-4 flex flex-col gap-4">
      <div className="flex items-center justify-between rounded-lg border border-border bg-card px-3.5 py-3">
        <div>
          <div className="font-mono text-[10px] uppercase tracking-wide text-dim">Trend</div>
          <div className="mt-1">
            <TrendBadge trend={data.trend} />
          </div>
        </div>
        <div className="text-right">
          <div className="font-mono text-[10px] uppercase tracking-wide text-dim">Moving Averages</div>
          <div className="mt-1 flex items-center justify-end gap-2">
            <VerdictBadge verdict={data.moving_average_signal.verdict} />
            <span className="font-mono text-[10px] text-dim">
              {data.moving_average_signal.buy} buy &middot; {data.moving_average_signal.sell} sell
            </span>
          </div>
        </div>
      </div>

      <div>
        <p className="mb-2 font-mono text-[10px] tracking-wide text-dim">INDICATORS</p>
        <div className="rounded-lg border border-border bg-card px-3.5 py-3">
          <StatGrid
            columns={3}
            items={[
              { label: "RSI (14)", value: fmtNum(ind.rsi_14, 1) },
              { label: "MACD", value: fmtNum(ind.macd.macd, 2) },
              { label: "MACD Signal", value: fmtNum(ind.macd.signal, 2) },
              { label: "ADX (14)", value: fmtNum(ind.adx_14, 1) },
              { label: "VWAP", value: fmtNum(ind.vwap, 2) },
              { label: "EMA 20 / 50 / 200", value: `${fmtNum(ind.ema_20, 1)} / ${fmtNum(ind.ema_50, 1)} / ${fmtNum(ind.ema_200, 1)}` },
            ]}
          />
        </div>
      </div>

      <div>
        <p className="mb-2 font-mono text-[10px] tracking-wide text-dim">OSCILLATORS</p>
        <div className="rounded-lg border border-border bg-card px-3.5 py-3">
          <StatGrid
            columns={3}
            items={[
              { label: "RSI (14)", value: fmtNum(ind.rsi_14, 1) },
              { label: "Stochastic %K", value: fmtNum(ind.stochastic.k, 1) },
              { label: "Stochastic %D", value: fmtNum(ind.stochastic.d, 1) },
              { label: "Williams %R", value: fmtNum(ind.williams_r, 1) },
            ]}
          />
        </div>
      </div>

      <div>
        <p className="mb-2 font-mono text-[10px] tracking-wide text-dim">
          SUPPORT / RESISTANCE <span className="normal-case text-dim">(recent swing levels, not vendor pivot points)</span>
        </p>
        <div className="rounded-lg border border-border bg-card px-3.5 py-3">
          <StatGrid
            columns={2}
            items={[
              { label: "Support (20d)", value: fmtPrice(sr.support_20d, symbol) },
              { label: "Resistance (20d)", value: fmtPrice(sr.resistance_20d, symbol) },
              { label: "Support (60d)", value: fmtPrice(sr.support_60d, symbol) },
              { label: "Resistance (60d)", value: fmtPrice(sr.resistance_60d, symbol) },
            ]}
          />
        </div>
      </div>
    </div>
  );
}
