"use client";

import { useState } from "react";
import { currencySymbol } from "@/lib/currency";
import { fmtCompactMoney, fmtRatio } from "@/lib/stockFormat";

interface PeerRow {
  ticker: string;
  company_name: string | null;
  currency: string;
  market_cap: number | null;
  trailing_pe: number | null;
  revenue_cagr_3y: number | null;
  eps_cagr_3y: number | null;
}

function fmtCagr(v: number | null): string {
  if (v === null) return "N/A";
  return `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`;
}

// User-entered peer only -- never auto-discovered. This app has no
// peer-multiple/industry data source (see relative_valuation.py's own
// docstring), so an automatic "similar stocks" comparison would be
// guessing; the user picking the ticker keeps every number here real.
export default function PeerComparisonTable({ ticker }: { ticker: string }) {
  const [peerInput, setPeerInput] = useState("");
  const [rows, setRows] = useState<{ primary: PeerRow; peer: PeerRow } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function compare(e: React.FormEvent) {
    e.preventDefault();
    const peer = peerInput.trim();
    if (!peer || loading) return;
    setLoading(true);
    setError(null);
    setRows(null);
    try {
      const resp = await fetch(`/api/stock/${encodeURIComponent(ticker)}/peer-comparison?peer=${encodeURIComponent(peer)}`);
      const data = await resp.json();
      if (!resp.ok) {
        setError(data.message || "Couldn't compare against that ticker.");
        return;
      }
      setRows(data);
    } catch {
      setError("Couldn't compare against that ticker.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mt-4">
      <p className="mb-2 font-mono text-[10px] tracking-wide text-dim">PEER COMPARISON</p>
      <div className="rounded-lg border border-border bg-card px-3.5 py-3">
        <form onSubmit={compare} className="flex gap-2">
          <input
            type="text"
            value={peerInput}
            onChange={(e) => setPeerInput(e.target.value)}
            placeholder="peer ticker (e.g. MSFT)..."
            disabled={loading}
            autoComplete="off"
            className="min-w-0 flex-1 rounded-lg border border-border bg-bg px-3 py-2 font-mono text-xs text-text placeholder:text-muted focus:outline-none focus:border-accent disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={loading || !peerInput.trim()}
            className="rounded-lg border border-border bg-card px-3.5 py-2 font-mono text-xs font-bold text-muted hover:border-accent hover:text-accent disabled:opacity-50"
          >
            {loading ? "..." : "COMPARE"}
          </button>
        </form>
        {error && <p className="mt-2 font-mono text-[10px] text-danger">{error}</p>}

        {rows && (
          <div className="mt-3 overflow-x-auto">
            <table className="w-full min-w-[320px] font-mono text-xs">
              <thead>
                <tr className="text-[10px] uppercase tracking-wide text-dim">
                  <th className="pb-2 text-left font-normal">Metric</th>
                  <th className="pb-2 text-right font-normal">{rows.primary.ticker}</th>
                  <th className="pb-2 text-right font-normal">{rows.peer.ticker}</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-t border-border-subtle">
                  <td className="py-1.5 text-dim">Market Cap</td>
                  <td className="py-1.5 text-right text-text">{fmtCompactMoney(rows.primary.market_cap, currencySymbol(rows.primary.currency))}</td>
                  <td className="py-1.5 text-right text-text">{fmtCompactMoney(rows.peer.market_cap, currencySymbol(rows.peer.currency))}</td>
                </tr>
                <tr className="border-t border-border-subtle">
                  <td className="py-1.5 text-dim">P/E</td>
                  <td className="py-1.5 text-right text-text">{fmtRatio(rows.primary.trailing_pe)}</td>
                  <td className="py-1.5 text-right text-text">{fmtRatio(rows.peer.trailing_pe)}</td>
                </tr>
                <tr className="border-t border-border-subtle">
                  <td className="py-1.5 text-dim">Revenue CAGR (3y)</td>
                  <td className="py-1.5 text-right text-text">{fmtCagr(rows.primary.revenue_cagr_3y)}</td>
                  <td className="py-1.5 text-right text-text">{fmtCagr(rows.peer.revenue_cagr_3y)}</td>
                </tr>
                <tr className="border-t border-border-subtle">
                  <td className="py-1.5 text-dim">EPS CAGR (3y)</td>
                  <td className="py-1.5 text-right text-text">{fmtCagr(rows.primary.eps_cagr_3y)}</td>
                  <td className="py-1.5 text-right text-text">{fmtCagr(rows.peer.eps_cagr_3y)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
