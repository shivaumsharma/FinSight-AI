import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { proxyFetch } from "@/lib/proxyFetch";
import { getSessionToken } from "@/lib/session";

// Proxies DELETE /v1/portfolio/{ticker}. params is a Promise in this
// Next.js version -- same shape as the watchlist/[ticker] sibling route.
export async function DELETE(
  request: NextRequest,
  context: { params: Promise<{ ticker: string }> }
) {
  const { ticker } = await context.params;
  const sessionToken = await getSessionToken();

  const resp = await proxyFetch(`${FINSIGHT_API_URL}/v1/portfolio/${ticker}`, {
    method: "DELETE",
    headers: backendHeaders(sessionToken),
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
