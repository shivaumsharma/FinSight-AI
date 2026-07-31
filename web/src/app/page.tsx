"use client";

import { useState } from "react";
import { useResearch } from "@/lib/useResearch";
import ReportView from "@/components/ReportView";

export default function Home() {
  const [query, setQuery] = useState("");
  const { status, jobId, result, errorMessage, latencySeconds, submit } = useResearch();

  const isBusy = status === "submitting" || status === "running";

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-black">
      <main className="mx-auto max-w-3xl px-6 py-16">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-gray-100">Finsight AI</h1>
        <p className="mt-1 text-lg text-gray-600 dark:text-gray-400">Autonomous Financial Intelligence Platform</p>

        <p className="mt-6 text-gray-700 dark:text-gray-300">
          FinSight AI is an Autonomous Financial Intelligence Platform designed for institutional-style equity
          research and investment analysis. It specializes in:
        </p>
        <ul className="mt-3 list-disc space-y-1 pl-6 text-gray-700 dark:text-gray-300">
          <li>Financial Statement Analysis</li>
          <li>Equity Research</li>
          <li>Intrinsic Valuation</li>
          <li>Earnings Call &amp; Filing Analysis</li>
          <li>Market Intelligence</li>
          <li>Investment Thesis Generation</li>
        </ul>

        <p className="mt-4 text-sm text-gray-500 dark:text-gray-400">
          <strong>FinSight AI is not a general-purpose financial chatbot.</strong> Please include a publicly listed
          company as part of your query.
        </p>
        <p className="mt-1 text-sm italic text-gray-500 dark:text-gray-400">
          Examples: &quot;Should I buy Apple?&quot; · &quot;Analyze NVIDIA&apos;s financial health.&quot; ·
          &quot;Generate a report on Microsoft.&quot; · &quot;Compare Google and Amazon.&quot;
        </p>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (query.trim() && !isBusy) submit(query.trim());
          }}
          className="mt-6 flex gap-2"
        >
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Should I buy Apple?"
            disabled={isBusy}
            className="flex-1 rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-gray-900 placeholder:text-gray-400 focus:border-gray-500 focus:outline-none disabled:opacity-60 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-100"
          />
          <button
            type="submit"
            disabled={isBusy || !query.trim()}
            className="rounded-lg bg-gray-900 px-6 py-2.5 font-semibold text-white hover:bg-gray-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-gray-100 dark:text-gray-900 dark:hover:bg-white"
          >
            {isBusy ? "Running..." : "Run Research Agent"}
          </button>
        </form>

        {isBusy && (
          <div className="mt-6 flex items-center gap-3 text-gray-600 dark:text-gray-400">
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-gray-400 border-t-transparent" />
            Planning and executing research...
          </div>
        )}

        {status === "error" && errorMessage && (
          <div className="mt-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-300">
            {errorMessage}
          </div>
        )}

        {status === "done" && result && jobId && (
          <ReportView result={result} jobId={jobId} latencySeconds={latencySeconds} />
        )}

        <p className="mt-16 text-center text-xs text-gray-400 dark:text-gray-600">
          Built using an LLM Planner + RAG over live SEC filings, ChromaDB, FinBERT and DCF valuation tools
        </p>
      </main>
    </div>
  );
}
