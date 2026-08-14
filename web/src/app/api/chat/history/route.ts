import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { proxyFetch } from "@/lib/proxyFetch";
import { getSessionToken } from "@/lib/session";

// Proxies GET /v1/chat/history.
export async function GET(request: NextRequest) {
  const sessionToken = await getSessionToken();
  const limit = request.nextUrl.searchParams.get("limit");

  const url = new URL(`${FINSIGHT_API_URL}/v1/chat/history`);
  if (limit) url.searchParams.set("limit", limit);

  const resp = await proxyFetch(url.toString(), {
    headers: backendHeaders(sessionToken),
    cache: "no-store",
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
