import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { proxyFetch } from "@/lib/proxyFetch";
import { getSessionToken } from "@/lib/session";

// Proxies POST /v1/research/{job_id}/what-if. Body is an optional
// {growth_rate_pct, wacc_pct, terminal_growth_pct} object -- an empty
// body ({}) on first load is valid, the backend fills in
// defaults/bounds for any field that's missing/null.
export async function POST(
  request: NextRequest,
  context: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await context.params;
  const sessionToken = await getSessionToken();
  const body = await request.json().catch(() => ({}));

  const resp = await proxyFetch(`${FINSIGHT_API_URL}/v1/research/${jobId}/what-if`, {
    method: "POST",
    headers: backendHeaders(sessionToken),
    body: JSON.stringify({
      growth_rate_pct: body.growth_rate_pct ?? null,
      wacc_pct: body.wacc_pct ?? null,
      terminal_growth_pct: body.terminal_growth_pct ?? null,
    }),
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
