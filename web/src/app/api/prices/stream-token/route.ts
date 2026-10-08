import { NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { proxyFetch } from "@/lib/proxyFetch";
import { getSessionToken } from "@/lib/session";

// Proxies GET /v1/prices/stream-token and adds the WebSocket URL the browser must connect to directly (Next.js route
// handlers cannot proxy WebSockets). PRICE_STREAM_WS_URL overrides the URL derived from FINSIGHT_API_URL; set it when
// the backend is reached over WebSocket through a different host (for example the CloudFront domain, as wss://).
export async function GET() {
  const sessionToken = await getSessionToken();

  const resp = await proxyFetch(`${FINSIGHT_API_URL}/v1/prices/stream-token`, {
    headers: backendHeaders(sessionToken),
    cache: "no-store",
  });

  if (!resp.ok) {
    const data = await resp.json().catch(() => ({ code: "STREAM_TOKEN_FAILED", message: resp.statusText }));
    return NextResponse.json(data, { status: resp.status });
  }

  const data = await resp.json();
  const wsUrl = process.env.PRICE_STREAM_WS_URL || `${FINSIGHT_API_URL.replace(/^http/, "ws")}/v1/prices/stream`;
  return NextResponse.json({ ...data, ws_url: wsUrl });
}
