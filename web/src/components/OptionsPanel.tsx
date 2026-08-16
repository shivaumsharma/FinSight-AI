"use client";

import { useCallback, useEffect, useState } from "react";
import { currencySymbol } from "@/lib/currency";
import { fmtPercent, fmtPrice } from "@/lib/stockFormat";
import type { ApiErrorBody, OptionContractRow, OptionsAnalysis } from "@/lib/types";

// Codes the backend returns for "no options data here" -- both are
// expected, common outcomes (most tickers, especially non-US ones,
// simply have no listed options market; see
// app/derivatives/options_pricer.py's OptionsUnavailableError and
// app/api/main.py's get_stock_options for exactly when each fires),
// so both render as a calm informational message, never the red error
// state reserved for a real backend/network failure. TICKER_NOT_FOUND
// is included even though this panel only ever sees a ticker the page
// itself already resolved -- it's still part of this endpoint's
// documented failure contract, so it's handled the same way here.
const OPTIONS_UNAVAILABLE_CODE = "OPTIONS_UNAVAILABLE";
const TICKER_NOT_FOUND_CODE = "TICKER_NOT_FOUND";

function fmtCell(v: number | null | undefined, decimals = 2): string {
  if (v === null || v === undefined) return "—";
  return v.toFixed(decimals);
}

function fmtMoneyCell(v: number | null | undefined, symbol: string): string {
  if (v === null || v === undefined) return "—";
  return `${symbol}${v.toFixed(2)}`;
}

function fmtOiCell(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return v.toLocaleString();
}

function ContractTable({ rows, symbol, label }: { rows: OptionContractRow[]; symbol: string; label: string }) {
  return (
    <div>
      <p className="mb-2 font-mono text-[10px] tracking-wide text-dim">{label}</p>
      {rows.length === 0 ? (
        <div className="rounded-lg border border-border-subtle bg-card/60 px-3.5 py-4 text-center">
          <p className="font-mono text-[11px] text-dim">No {label.toLowerCase()} contracts for this expiry.</p>
        </div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-border bg-card">
          <table className="w-full min-w-[760px] font-mono text-[11px]">
            <thead>
              <tr className="text-[10px] uppercase tracking-wide text-dim">
                <th className="px-2.5 py-2 text-left font-normal">Strike</th>
                <th className="px-2.5 py-2 text-right font-normal">Mkt Price</th>
                <th className="px-2.5 py-2 text-right font-normal">IV%</th>
                <th className="px-2.5 py-2 text-right font-normal">Theo Price</th>
                <th className="px-2.5 py-2 text-right font-normal">Delta</th>
                <th className="px-2.5 py-2 text-right font-normal">Gamma</th>
                <th className="px-2.5 py-2 text-right font-normal">Theta</th>
                <th className="px-2.5 py-2 text-right font-normal">Vega</th>
                <th className="px-2.5 py-2 text-right font-normal">Rho</th>
                <th className="px-2.5 py-2 text-right font-normal">OI</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.strike} className={`border-t border-border-subtle ${r.in_the_money ? "bg-accent/5" : ""}`}>
                  <td className="px-2.5 py-1.5 text-left text-text">{fmtMoneyCell(r.strike, symbol)}</td>
                  <td className="px-2.5 py-1.5 text-right text-text">{fmtMoneyCell(r.market_price, symbol)}</td>
                  <td className="px-2.5 py-1.5 text-right text-text">
                    {r.implied_vol_pct === null ? "—" : `${r.implied_vol_pct.toFixed(1)}%`}
                  </td>
                  <td className="px-2.5 py-1.5 text-right text-text">{fmtMoneyCell(r.theoretical_price, symbol)}</td>
                  <td className="px-2.5 py-1.5 text-right text-text">{fmtCell(r.delta, 3)}</td>
                  <td className="px-2.5 py-1.5 text-right text-text">{fmtCell(r.gamma, 4)}</td>
                  <td className="px-2.5 py-1.5 text-right text-text">{fmtCell(r.theta, 3)}</td>
                  <td className="px-2.5 py-1.5 text-right text-text">{fmtCell(r.vega, 3)}</td>
                  <td className="px-2.5 py-1.5 text-right text-text">{fmtCell(r.rho, 3)}</td>
                  <td className="px-2.5 py-1.5 text-right text-dim">{fmtOiCell(r.open_interest)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// Self-fetches /options independently, same convention as
// TechnicalsPanel/EventsTab -- loads on ticker change, and again
// whenever the user picks a different expiry from the dropdown below.
// loadOptions is keyed off `ticker` via useCallback so the expiry-
// change handler and the ticker-change effect can share one fetch path
// without either one going stale.
export default function OptionsPanel({ ticker }: { ticker: string }) {
  const [data, setData] = useState<OptionsAnalysis | null>(null);
  const [noOptions, setNoOptions] = useState<string | null>(null);
  const [error, setError] = useState(false);

  const loadOptions = useCallback(
    (expiry: string | null) => {
      setData(null);
      setNoOptions(null);
      setError(false);
      const qs = expiry ? `?expiry=${encodeURIComponent(expiry)}` : "";
      fetch(`/api/stock/${encodeURIComponent(ticker)}/options${qs}`)
        .then(async (r) => {
          const body = await r.json();
          if (!r.ok) {
            const errBody = body as ApiErrorBody;
            if (errBody.code === OPTIONS_UNAVAILABLE_CODE || errBody.code === TICKER_NOT_FOUND_CODE) {
              setNoOptions(errBody.message || "No listed options available for this ticker.");
            } else {
              setError(true);
            }
            return;
          }
          setData(body as OptionsAnalysis);
        })
        .catch(() => setError(true));
    },
    [ticker]
  );

  useEffect(() => {
    loadOptions(null);
  }, [loadOptions]);

  if (error) {
    return <p className="mt-4 py-6 text-center font-mono text-[11px] text-dim">Couldn&apos;t load options data for this ticker.</p>;
  }

  if (noOptions) {
    return (
      <div className="mt-4 rounded-lg border border-border-subtle bg-card/60 px-3.5 py-6 text-center">
        <p className="font-mono text-[11px] text-dim">{noOptions}</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="mt-4 flex flex-col gap-2">
        <div className="h-[60px] animate-pulse rounded-lg bg-card/60" />
        <div className="h-[220px] animate-pulse rounded-lg bg-card/60" />
      </div>
    );
  }

  const symbol = currencySymbol(data.currency);

  return (
    <div className="mt-4 flex flex-col gap-4">
      <div className="rounded-lg border border-amber-900/60 bg-amber-950/40 px-3.5 py-2.5 text-[11px] text-warn">
        Theoretical Black-Scholes pricing computed from live market inputs &mdash; informational only, not a trading
        signal or investment advice.
      </div>

      <div className="rounded-lg border border-border bg-card px-3.5 py-3">
        <div className="grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-4">
          <div>
            <div className="font-mono text-[10px] uppercase tracking-wide text-dim">Spot Price</div>
            <div className="font-mono text-sm text-text">{fmtPrice(data.spot_price, symbol)}</div>
          </div>
          <div>
            <div className="font-mono text-[10px] uppercase tracking-wide text-dim">Risk-Free Rate</div>
            <div className="font-mono text-sm text-text">{fmtPercent(data.risk_free_rate, true)}</div>
          </div>
          <div>
            <div className="font-mono text-[10px] uppercase tracking-wide text-dim">Realized Vol</div>
            <div className="font-mono text-sm text-text">{fmtPercent(data.realized_volatility_pct)}</div>
          </div>
          <div>
            <div className="font-mono text-[10px] uppercase tracking-wide text-dim">Days to Expiry</div>
            <div className="font-mono text-sm text-text">{data.days_to_expiry}</div>
          </div>
        </div>

        <div className="mt-3 border-t border-border-subtle pt-3">
          <label className="font-mono text-[10px] uppercase tracking-wide text-dim" htmlFor="options-expiry-select">
            Expiry
          </label>
          <select
            id="options-expiry-select"
            value={data.selected_expiry}
            onChange={(e) => loadOptions(e.target.value)}
            className="mt-1 w-full rounded-lg border border-border bg-bg px-3 py-2 font-mono text-xs text-text focus:outline-none focus:border-accent"
          >
            {data.expiries.map((exp) => (
              <option key={exp} value={exp}>
                {exp}
              </option>
            ))}
          </select>
        </div>
      </div>

      <ContractTable rows={data.calls} symbol={symbol} label="Calls" />
      <ContractTable rows={data.puts} symbol={symbol} label="Puts" />
    </div>
  );
}
