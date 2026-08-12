import StatGrid from "./StatGrid";
import { currencySymbol } from "@/lib/currency";
import { fmtCompactMoney, fmtPercent, fmtPrice, fmtRatio } from "@/lib/stockFormat";
import type { StockOverview } from "@/lib/types";

// No ROCE, Industry P/E, or Promoter Holding here -- yfinance has no
// industry-aggregate P/E concept and no promoter-holding data at all
// (Indian-market-specific, paid-vendor-only); see the approved
// stock-detail-page plan's "Explicitly out of scope this pass". Every
// field below is real, wired data, not a placeholder.
export default function FundamentalsCard({ overview }: { overview: StockOverview }) {
  const symbol = currencySymbol(overview.currency);
  const f = overview.fundamentals;
  const a = overview.analyst;

  return (
    <div className="mt-4">
      <p className="mb-2 font-mono text-[10px] tracking-wide text-dim">FUNDAMENTALS</p>
      <div className="rounded-lg border border-border bg-card px-3.5 py-3">
        <StatGrid
          columns={3}
          items={[
            { label: "Market Cap", value: fmtCompactMoney(overview.market_cap, symbol) },
            { label: "P/E (Trailing)", value: fmtRatio(f.trailing_pe) },
            { label: "P/E (Forward)", value: fmtRatio(f.forward_pe) },
            { label: "P/B", value: fmtRatio(f.price_to_book) },
            { label: "EPS (TTM)", value: fmtPrice(f.trailing_eps, symbol) },
            { label: "Dividend Yield", value: fmtPercent(f.dividend_yield) },
            { label: "Book Value", value: fmtPrice(f.book_value, symbol) },
            { label: "Debt to Equity", value: fmtRatio(f.debt_to_equity) },
            { label: "PEG Ratio", value: fmtRatio(f.peg_ratio) },
          ]}
        />
        {a.target_mean_price !== null && (
          <div className="mt-3 border-t border-border-subtle pt-3">
            <StatGrid
              columns={3}
              items={[
                { label: "Analyst Target (Avg)", value: fmtPrice(a.target_mean_price, symbol) },
                { label: "Target High", value: fmtPrice(a.target_high_price, symbol) },
                { label: "Target Low", value: fmtPrice(a.target_low_price, symbol) },
              ]}
            />
            {a.number_of_analyst_opinions !== null && (
              <p className="mt-2 font-mono text-[10px] text-dim">
                Based on {a.number_of_analyst_opinions} analyst opinion{a.number_of_analyst_opinions === 1 ? "" : "s"}.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
