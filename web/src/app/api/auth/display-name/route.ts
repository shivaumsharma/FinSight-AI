import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { proxyFetch } from "@/lib/proxyFetch";
import { getSessionToken } from "@/lib/session";

// Proxies PATCH /v1/auth/display-name.
export async function PATCH(request: NextRequest) {
  const body = await request.json();
  const sessionToken = await getSessionToken();

  const resp = await proxyFetch(`${FINSIGHT_API_URL}/v1/auth/display-name`, {
    method: "PATCH",
    headers: backendHeaders(sessionToken),
    body: JSON.stringify({ display_name: body.display_name }),
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
