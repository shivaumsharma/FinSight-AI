import { NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { getSessionToken } from "@/lib/session";

// Proxies GET /v1/voice/wake-listen-token, then adds the one thing
// only this server-side route can compute: the backend's own public
// WebSocket URL, derived from FINSIGHT_API_URL (server-only, never
// otherwise sent to the browser -- see config.ts's own comment). The
// browser needs this because the wake-word listener's WebSocket has
// to connect to Cloud Run DIRECTLY, not through this Next.js app (see
// app/api/main.py's wake_listen endpoint docstring for why: Next.js
// Route Handlers cannot proxy WebSockets at all).
export async function GET() {
  const sessionToken = await getSessionToken();

  const resp = await fetch(`${FINSIGHT_API_URL}/v1/voice/wake-listen-token`, {
    headers: backendHeaders(sessionToken),
  });

  if (!resp.ok) {
    const data = await resp.json().catch(() => ({ code: "WAKE_TOKEN_FETCH_FAILED", message: resp.statusText }));
    return NextResponse.json(data, { status: resp.status });
  }

  const data = await resp.json();
  const wsUrl = `${FINSIGHT_API_URL.replace(/^http/, "ws")}/v1/voice/wake-listen`;
  return NextResponse.json({ ...data, ws_url: wsUrl });
}
