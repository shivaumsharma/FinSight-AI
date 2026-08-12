// Shared label/value grid for the stock-detail-page cards
// (PriceStatistics/FundamentalsCard/GrowthMetrics/...) -- same visual
// shape as ReportView.tsx's own KeyValueGrid, but pre-formatted string
// values are passed in directly rather than this component doing any
// field-name-based formatting itself (that logic lives in
// lib/stockFormat.ts, reused across cards with very different field
// sets). "N/A" entries still render (greyed) rather than being
// filtered out -- an investor scanning a fundamentals card should see
// which fields yfinance simply doesn't have for this ticker, not have
// them silently vanish.
export interface StatItem {
  label: string;
  value: string;
}

export default function StatGrid({ items, columns = 2 }: { items: StatItem[]; columns?: 2 | 3 }) {
  return (
    <div className={`grid gap-x-6 gap-y-3 ${columns === 3 ? "grid-cols-2 sm:grid-cols-3" : "grid-cols-2"}`}>
      {items.map((item) => (
        <div key={item.label}>
          <div className="font-mono text-[10px] uppercase tracking-wide text-dim">{item.label}</div>
          <div className={`font-mono text-sm ${item.value === "N/A" ? "text-dim" : "text-text"}`}>{item.value}</div>
        </div>
      ))}
    </div>
  );
}
