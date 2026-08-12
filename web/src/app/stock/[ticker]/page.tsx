"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";
import AIInsightsCard from "@/components/AIInsightsCard";
import AuthGate from "@/components/AuthGate";
import BottomNav from "@/components/BottomNav";
import CompanyInfoCard from "@/components/CompanyInfoCard";
import CompanyOverviewCard from "@/components/CompanyOverviewCard";
import EventsTab from "@/components/EventsTab";
import FinancialPerformanceChart from "@/components/FinancialPerformanceChart";
import FundamentalsCard from "@/components/FundamentalsCard";
import GrowthMetrics from "@/components/GrowthMetrics";
import ModelCompare from "@/components/ModelCompare";
import NewsTab from "@/components/NewsTab";
import PeerComparisonTable from "@/components/PeerComparisonTable";
import PerformanceDashboard from "@/components/PerformanceDashboard";
import PriceChart from "@/components/PriceChart";
import PriceStatistics from "@/components/PriceStatistics";
import RecentlyViewed from "@/components/RecentlyViewed";
import SimilarStocks from "@/components/SimilarStocks";
import StockHeader from "@/components/StockHeader";
import Tabs from "@/components/Tabs";
import TechnicalsPanel from "@/components/TechnicalsPanel";
import TradeBar from "@/components/TradeBar";
import type { ApiErrorBody, StockOverview } from "@/lib/types";

// Canonical per-ticker page -- Watchlist/MarketMovers/Reports/Portfolio
// all link here now instead of re-running research just to see a
// price. The existing research-report flow (ReportView.tsx) is
// untouched and reachable below as "Run Full Research Report" -- this
// page is additive, not a replacement.
//
// params is a Promise in this Next.js version (see every proxy route's
// own comment on the same point); `use()` is the documented way to
// unwrap it in a Client Component page.
export default function StockDetailPage({ params }: { params: Promise<{ ticker: string }> }) {
  const { ticker } = use(params);
  return <AuthGate>{() => <StockDetailContent ticker={ticker} />}</AuthGate>;
}

function StockDetailContent({ ticker }: { ticker: string }) {
  const [overview, setOverview] = useState<StockOverview | null>(null);
  const [error, setError] = useState<ApiErrorBody | null>(null);

  useEffect(() => {
    setOverview(null);
    setError(null);
    fetch(`/api/stock/${encodeURIComponent(ticker)}/overview`)
      .then(async (r) => {
        const data = await r.json();
        if (!r.ok) {
          setError(data);
          return;
        }
        setOverview(data);
      })
      .catch(() => setError({ code: "NETWORK_ERROR", message: "Couldn't load this stock. Check your connection and try again." }));
  }, [ticker]);

  if (error) {
    return (
      <div className="min-h-screen bg-bg pb-20">
        <div className="mx-auto max-w-2xl px-5 py-8">
          <Link href="/" className="font-mono text-[10px] font-bold text-dim hover:text-accent">
            &larr; BACK
          </Link>
          <div className="mt-6 rounded-lg border border-red-900/60 bg-red-950/40 px-4 py-3 text-sm text-danger">
            {error.message || "Couldn't load this stock."}
          </div>
        </div>
        <BottomNav />
      </div>
    );
  }

  if (!overview) {
    return (
      <div className="min-h-screen bg-bg pb-20">
        <div className="mx-auto max-w-2xl px-5 py-8">
          <p className="font-mono text-xs text-dim">loading...</p>
        </div>
        <BottomNav />
      </div>
    );
  }

  return (
    // pb-36, not pb-20 like every other page -- this one stacks TWO
    // fixed bottom bars (TradeBar above BottomNav), so needs extra
    // clearance to keep the last card from being covered by either.
    <div className="min-h-screen bg-bg pb-36">
      <div className="mx-auto max-w-2xl px-5 py-8">
        <StockHeader overview={overview} />

        <div className="mt-4 rounded-lg border border-amber-900/60 bg-amber-950/40 px-3.5 py-2.5 text-[11px] text-warn">
          Informational data only, not personalized investment advice. Consult a licensed financial advisor before
          making investment decisions.
        </div>

        <PerformanceDashboard ticker={overview.ticker} />

        <div className="mt-6">
          <Tabs
            tabs={[
              {
                label: "Overview",
                content: (
                  <div>
                    <AIInsightsCard ticker={overview.ticker} currency={overview.currency} />
                    <div className="mt-4 rounded-lg border border-border bg-card px-3.5 py-3">
                      <p className="text-xs text-muted">Get a second opinion: 3 independent AI models each give their own Buy/Hold/Sell call on this stock.</p>
                      <ModelCompare endpoint={`/api/stock/${encodeURIComponent(overview.ticker)}/model-compare`} />
                    </div>
                    <PriceStatistics overview={overview} />
                    <FundamentalsCard overview={overview} />
                    <FinancialPerformanceChart ticker={overview.ticker} currency={overview.currency} />
                    <GrowthMetrics ticker={overview.ticker} />
                    <CompanyOverviewCard overview={overview} />
                    <CompanyInfoCard overview={overview} />
                    <PeerComparisonTable ticker={overview.ticker} />
                    <SimilarStocks ticker={overview.ticker} />
                    <RecentlyViewed ticker={overview.ticker} companyName={overview.company_name} />
                    <div className="mt-4 rounded-lg border border-border bg-card px-3.5 py-3 text-center">
                      <Link
                        href={`/?q=${encodeURIComponent(`Should I invest in ${overview.ticker}?`)}`}
                        className="font-mono text-xs font-bold text-accent hover:underline"
                      >
                        RUN FULL RESEARCH REPORT &rarr;
                      </Link>
                    </div>
                  </div>
                ),
              },
              {
                label: "Technicals",
                content: (
                  <div>
                    <PriceChart ticker={overview.ticker} currency={overview.currency} />
                    <TechnicalsPanel ticker={overview.ticker} currency={overview.currency} />
                  </div>
                ),
              },
              {
                label: "News",
                content: <NewsTab ticker={overview.ticker} />,
              },
              {
                label: "Events",
                content: <EventsTab ticker={overview.ticker} currency={overview.currency} />,
              },
            ]}
          />
        </div>
      </div>
      <TradeBar ticker={overview.ticker} currency={overview.currency} />
      <BottomNav />
    </div>
  );
}
