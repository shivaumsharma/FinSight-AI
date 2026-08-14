import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { proxyFetch } from "@/lib/proxyFetch";
import { getSessionToken } from "@/lib/session";

// Proxies GET /v1/market/movers -- same thin-proxy pattern as every
// other route in this directory.
export async function GET(request: NextRequest) {
  const limit = request.nextUrl.searchParams.get("limit") || "5";
  const sessionToken = await getSessionToken();

  const resp = await proxyFetch(`${FINSIGHT_API_URL}/v1/market/movers?limit=${encodeURIComponent(limit)}`, {
    headers: backendHeaders(sessionToken),
    cache: "no-store",
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
