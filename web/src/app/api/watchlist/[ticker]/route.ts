import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { getSessionToken } from "@/lib/session";

// Proxies DELETE /v1/watchlist/{ticker}. params is a Promise in this
// Next.js version (see the [jobId]/route.ts sibling route) -- same
// shape copied here, not assumed from memory.
export async function DELETE(
  request: NextRequest,
  context: { params: Promise<{ ticker: string }> }
) {
  const { ticker } = await context.params;
  const sessionToken = await getSessionToken();

  const resp = await fetch(`${FINSIGHT_API_URL}/v1/watchlist/${ticker}`, {
    method: "DELETE",
    headers: backendHeaders(sessionToken),
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
