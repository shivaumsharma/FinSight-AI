import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { proxyFetch } from "@/lib/proxyFetch";
import { getSessionToken } from "@/lib/session";

// Proxies GET/POST /v1/watchlist -- same thin-proxy pattern as every
// other route in this directory.
export async function GET() {
  const sessionToken = await getSessionToken();

  const resp = await proxyFetch(`${FINSIGHT_API_URL}/v1/watchlist`, {
    headers: backendHeaders(sessionToken),
    cache: "no-store",
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}

export async function POST(request: NextRequest) {
  const body = await request.json();
  const sessionToken = await getSessionToken();

  const resp = await proxyFetch(`${FINSIGHT_API_URL}/v1/watchlist`, {
    method: "POST",
    headers: backendHeaders(sessionToken),
    body: JSON.stringify({ ticker: body.ticker }),
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
