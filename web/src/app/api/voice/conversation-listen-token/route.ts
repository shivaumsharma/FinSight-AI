import { NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { proxyFetch } from "@/lib/proxyFetch";
import { getSessionToken } from "@/lib/session";

// Proxies GET /v1/voice/conversation-listen-token, then adds the
// backend's own public WebSocket URL, same reasoning as
// wake-listen-token/route.ts's own comment: Next.js Route Handlers
// cannot proxy a WebSocket at all, so the browser has to connect to
// the backend directly for this one.
export async function GET() {
  const sessionToken = await getSessionToken();

  const resp = await proxyFetch(`${FINSIGHT_API_URL}/v1/voice/conversation-listen-token`, {
    headers: backendHeaders(sessionToken),
  });

  if (!resp.ok) {
    const data = await resp.json().catch(() => ({ code: "CONVERSATION_TOKEN_FETCH_FAILED", message: resp.statusText }));
    return NextResponse.json(data, { status: resp.status });
  }

  const data = await resp.json();
  const wsUrl = `${FINSIGHT_API_URL.replace(/^http/, "ws")}/v1/voice/conversation-listen`;
  return NextResponse.json({ ...data, ws_url: wsUrl });
}
