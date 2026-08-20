import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { proxyFetch } from "@/lib/proxyFetch";
import { getSessionToken } from "@/lib/session";

// Proxies GET /v1/market/screener -- same thin-proxy pattern as every
// other route in this directory (e.g. market/movers). Forwards every
// filter/sort param through unchanged; the backend owns validation and
// defaults, this route just passes along whatever query string it got.
export async function GET(request: NextRequest) {
  const sessionToken = await getSessionToken();

  const resp = await proxyFetch(`${FINSIGHT_API_URL}/v1/market/screener?${request.nextUrl.searchParams.toString()}`, {
    headers: backendHeaders(sessionToken),
    cache: "no-store",
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
