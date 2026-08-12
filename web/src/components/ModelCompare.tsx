"use client";

import { useState } from "react";
import RatingBadge, { ratingColorClass } from "./RatingBadge";
import type { ModelConsensus, ModelOpinion } from "@/lib/types";

// On-demand second opinion: 3 independent models re-interpreting the
// SAME already-computed evidence (see app/reasoning/model_consensus.py),
// not a re-run of the full pipeline -- triggered by a button rather
// than run automatically, since it's extra LLM cost/latency a viewer
// may not want on every single query. `endpoint` is the full proxy
// path to POST to -- the job-based report flow and the ticker-based
// stock-detail-page flow hit different backend endpoints (no completed
// job exists on the stock page), so this component stays endpoint-
// agnostic rather than reconstructing the URL from a jobId/ticker prop.
export default function ModelCompare({ endpoint }: { endpoint: string }) {
  const [state, setState] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [models, setModels] = useState<ModelOpinion[] | null>(null);
  const [consensus, setConsensus] = useState<ModelConsensus | null>(null);

  async function handleCompare() {
    setState("loading");
    try {
      const resp = await fetch(endpoint, { method: "POST" });
      if (!resp.ok) throw new Error("model compare failed");
      const data = await resp.json();
      setModels(data.models);
      setConsensus(data.consensus);
      setState("done");
    } catch {
      setState("error");
    }
  }

  if (state === "idle" || state === "error") {
    return (
      <div className="mt-2">
        <button
          type="button"
          onClick={handleCompare}
          className="mt-6 ml-2 inline-block rounded-lg border border-border bg-card px-5 py-2.5 font-mono text-xs font-bold text-text hover:border-accent"
        >
          COMPARE MODELS
        </button>
        {state === "error" && <p className="mt-1.5 ml-2 font-mono text-[10px] text-danger">Couldn&apos;t reach the models -- try again.</p>}
      </div>
    );
  }

  if (state === "loading") {
    return (
      <div className="mt-6 ml-2 font-mono text-xs text-muted">Asking 3 independent models&hellip;</div>
    );
  }

  return (
    <div className="mt-8">
      <div className="mb-2 font-mono text-[11px] font-bold tracking-wide text-muted">MODEL COMPARE</div>

      {consensus && (
        <div className={`rounded-lg border bg-card px-4 py-2.5 text-center font-mono text-xs font-bold ${ratingColorClass(consensus.rating)}`}
             style={{ borderColor: consensus.rating === "Insufficient Data" ? "var(--border)" : "currentColor" }}>
          {consensus.rating === "Insufficient Data"
            ? "No model produced a usable opinion"
            : `${consensus.agree_count} / ${consensus.total} AGREE · ${consensus.rating.toUpperCase()}`}
        </div>
      )}

      <div className="mt-3 flex flex-col gap-2">
        {models?.map((m) => (
          <div key={m.model} className="rounded-lg border border-border bg-card px-3.5 py-3">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[11px] font-bold tracking-wide text-muted">{m.label.toUpperCase()}</span>
              <RatingBadge rating={m.rating} size="sm" />
            </div>
            {m.confidence !== null && (
              <div className="mt-1 font-mono text-[10px] text-dim">conf. {m.confidence}%</div>
            )}
            <p className="mt-1.5 text-xs text-text/90">{m.reasoning}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
