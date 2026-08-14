import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { proxyFetch } from "@/lib/proxyFetch";
import { getSessionToken } from "@/lib/session";

// Proxies GET /v1/stocks/{ticker}/financials?period=yearly|quarterly.
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ ticker: string }> }
) {
  const { ticker } = await context.params;
  const period = request.nextUrl.searchParams.get("period") || "yearly";
  const sessionToken = await getSessionToken();

  const resp = await proxyFetch(
    `${FINSIGHT_API_URL}/v1/stocks/${encodeURIComponent(ticker)}/financials?period=${encodeURIComponent(period)}`,
    { headers: backendHeaders(sessionToken), cache: "no-store" }
  );

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
