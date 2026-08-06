import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { getSessionToken } from "@/lib/session";

// Proxies GET/POST /v1/portfolio -- same thin-proxy pattern as
// web/src/app/api/watchlist/route.ts.
export async function GET() {
  const sessionToken = await getSessionToken();

  const resp = await fetch(`${FINSIGHT_API_URL}/v1/portfolio`, {
    headers: backendHeaders(sessionToken),
    cache: "no-store",
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}

export async function POST(request: NextRequest) {
  const body = await request.json();
  const sessionToken = await getSessionToken();

  const resp = await fetch(`${FINSIGHT_API_URL}/v1/portfolio`, {
    method: "POST",
    headers: backendHeaders(sessionToken),
    body: JSON.stringify({ ticker: body.ticker, quantity: body.quantity, avg_cost: body.avg_cost }),
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
