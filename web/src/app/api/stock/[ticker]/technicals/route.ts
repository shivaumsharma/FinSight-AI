import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { proxyFetch } from "@/lib/proxyFetch";
import { getSessionToken } from "@/lib/session";

// Proxies GET /v1/stocks/{ticker}/technicals?range=1mo|3mo|6mo|1y|2y|5y|max.
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ ticker: string }> }
) {
  const { ticker } = await context.params;
  const range = request.nextUrl.searchParams.get("range") || "1y";
  const sessionToken = await getSessionToken();

  const resp = await proxyFetch(
    `${FINSIGHT_API_URL}/v1/stocks/${encodeURIComponent(ticker)}/technicals?range=${encodeURIComponent(range)}`,
    { headers: backendHeaders(sessionToken), cache: "no-store" }
  );

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
