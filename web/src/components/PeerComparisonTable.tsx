"use client";

import { useState } from "react";
import { currencySymbol } from "@/lib/currency";
import { fmtCompactMoney, fmtRatio } from "@/lib/stockFormat";
import { formatPercent } from "@/lib/numberFormat";
import { Button, Table } from "./ui";

interface PeerRow {
  ticker: string;
  company_name: string | null;
  currency: string;
  market_cap: number | null;
  trailing_pe: number | null;
  revenue_cagr_3y: number | null;
  eps_cagr_3y: number | null;
}

const fmtCagr = (v: number | null): string => formatPercent(v, { decimals: 1, signed: true });

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
          <Button type="submit" disabled={!peerInput.trim()} loading={loading}>
            {loading ? "..." : "COMPARE"}
          </Button>
        </form>
        {error && <p className="mt-2 font-mono text-[10px] text-danger">{error}</p>}

        {rows && (
          <div className="mt-3">
            <Table
              caption={`${rows.primary.ticker} compared with ${rows.peer.ticker}`}
              columns={["Metric", rows.primary.ticker, rows.peer.ticker]}
              rows={[
                { key: "mcap", label: "Market Cap", cells: [fmtCompactMoney(rows.primary.market_cap, currencySymbol(rows.primary.currency)), fmtCompactMoney(rows.peer.market_cap, currencySymbol(rows.peer.currency))] },
                { key: "pe", label: "P/E", cells: [fmtRatio(rows.primary.trailing_pe), fmtRatio(rows.peer.trailing_pe)] },
                { key: "rev", label: "Revenue CAGR (3y)", cells: [fmtCagr(rows.primary.revenue_cagr_3y), fmtCagr(rows.peer.revenue_cagr_3y)] },
                { key: "eps", label: "EPS CAGR (3y)", cells: [fmtCagr(rows.primary.eps_cagr_3y), fmtCagr(rows.peer.eps_cagr_3y)] },
              ]}
            />
          </div>
        )}
      </div>
    </div>
  );
}
