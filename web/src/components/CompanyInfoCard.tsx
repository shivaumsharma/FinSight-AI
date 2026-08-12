import { fmtCompactNumber } from "@/lib/stockFormat";
import type { StockOverview } from "@/lib/types";

// No CEO/Founded/Auditor/Registrar/ISIN/Listing Date -- yfinance's
// `.info` has none of these for either US or NSE tickers; see the
// approved stock-detail-page plan's "Explicitly out of scope this
// pass". Only real, wired fields render, and each is individually
// omitted (not shown as "N/A") when yfinance didn't have it for this
// particular ticker.
export default function CompanyInfoCard({ overview }: { overview: StockOverview }) {
  const rows: { label: string; value: string }[] = [];
  if (overview.exchange) rows.push({ label: "Exchange", value: overview.exchange });
  if (overview.country) rows.push({ label: "Country", value: overview.country });
  if (overview.employees) rows.push({ label: "Employees", value: fmtCompactNumber(overview.employees) });
  if (overview.website) rows.push({ label: "Website", value: overview.website.replace(/^https?:\/\//, "") });

  if (rows.length === 0) return null;

  return (
    <div className="mt-4">
      <p className="mb-2 font-mono text-[10px] tracking-wide text-dim">COMPANY INFORMATION</p>
      <div className="rounded-lg border border-border bg-card px-3.5 py-3">
        <div className="flex flex-col gap-2">
          {rows.map((r) => (
            <div key={r.label} className="flex items-center justify-between">
              <span className="font-mono text-[10px] uppercase tracking-wide text-dim">{r.label}</span>
              {r.label === "Website" ? (
                <a
                  href={overview.website!}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="truncate font-mono text-xs text-accent hover:underline"
                >
                  {r.value}
                </a>
              ) : (
                <span className="font-mono text-xs text-text">{r.value}</span>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
