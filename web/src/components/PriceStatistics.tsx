import StatGrid from "./StatGrid";
import { currencySymbol } from "@/lib/currency";
import { fmtCompactNumber, fmtPrice } from "@/lib/stockFormat";
import type { StockOverview } from "@/lib/types";

// No upper/lower circuit limits here -- those are NSE-specific and not
// exposed by yfinance for any ticker (see the approved stock-detail-page
// plan's "Explicitly out of scope this pass"); showing a blank/fake
// value would be worse than not claiming the field at all.
export default function PriceStatistics({ overview }: { overview: StockOverview }) {
  const symbol = currencySymbol(overview.currency);
  const s = overview.price_statistics;

  return (
    <div className="mt-4">
      <p className="mb-2 font-mono text-[10px] tracking-wide text-dim">PRICE STATISTICS</p>
      <div className="rounded-lg border border-border bg-card px-3.5 py-3">
        <StatGrid
          columns={3}
          items={[
            { label: "Open", value: fmtPrice(s.open, symbol) },
            { label: "Previous Close", value: fmtPrice(overview.previous_close, symbol) },
            { label: "Day High", value: fmtPrice(s.day_high, symbol) },
            { label: "Day Low", value: fmtPrice(s.day_low, symbol) },
            { label: "52W High", value: fmtPrice(s.fifty_two_week_high, symbol) },
            { label: "52W Low", value: fmtPrice(s.fifty_two_week_low, symbol) },
            { label: "Volume", value: fmtCompactNumber(s.volume) },
            { label: "Avg Volume", value: fmtCompactNumber(s.average_volume) },
          ]}
        />
      </div>
    </div>
  );
}
