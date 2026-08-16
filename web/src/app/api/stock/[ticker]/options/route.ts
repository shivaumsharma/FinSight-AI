import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { proxyFetch } from "@/lib/proxyFetch";
import { getSessionToken } from "@/lib/session";

// Proxies GET /v1/stocks/{ticker}/options?expiry=YYYY-MM-DD (expiry optional).
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ ticker: string }> }
) {
  const { ticker } = await context.params;
  const expiry = request.nextUrl.searchParams.get("expiry");
  const sessionToken = await getSessionToken();

  const qs = expiry ? `?expiry=${encodeURIComponent(expiry)}` : "";
  const resp = await proxyFetch(
    `${FINSIGHT_API_URL}/v1/stocks/${encodeURIComponent(ticker)}/options${qs}`,
    { headers: backendHeaders(sessionToken), cache: "no-store" }
  );

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
