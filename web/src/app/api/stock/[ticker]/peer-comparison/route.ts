import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { proxyFetch } from "@/lib/proxyFetch";
import { getSessionToken } from "@/lib/session";

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ ticker: string }> }
) {
  const { ticker } = await context.params;
  const peer = request.nextUrl.searchParams.get("peer") || "";
  const sessionToken = await getSessionToken();

  const resp = await proxyFetch(
    `${FINSIGHT_API_URL}/v1/stocks/${encodeURIComponent(ticker)}/peer-comparison?peer=${encodeURIComponent(peer)}`,
    { headers: backendHeaders(sessionToken), cache: "no-store" }
  );

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
