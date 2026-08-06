import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { getSessionToken } from "@/lib/session";

// Proxies GET /v1/companies/suggest -- same thin-proxy pattern as
// every other route in this directory.
export async function GET(request: NextRequest) {
  const q = request.nextUrl.searchParams.get("q") || "";
  const sessionToken = await getSessionToken();

  const resp = await fetch(`${FINSIGHT_API_URL}/v1/companies/suggest?q=${encodeURIComponent(q)}`, {
    headers: backendHeaders(sessionToken),
    cache: "no-store",
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
