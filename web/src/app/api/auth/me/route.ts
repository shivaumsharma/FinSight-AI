import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { getSessionToken, clearSessionCookie } from "@/lib/session";

// Proxies GET /v1/auth/me -- the client calls this once on load to
// decide whether to show the research UI or the login/signup gate.
export async function GET() {
  const sessionToken = await getSessionToken();

  // Short-circuits without a backend round-trip when there's plainly no
  // cookie -- the overwhelmingly common case for a first-time or
  // logged-out visitor, and avoids depending on the backend being
  // reachable just to say "you're not logged in."
  if (!sessionToken) {
    return NextResponse.json({ code: "UNAUTHORIZED", message: "Not logged in." }, { status: 401 });
  }

  const resp = await fetch(`${FINSIGHT_API_URL}/v1/auth/me`, {
    headers: backendHeaders(sessionToken),
    cache: "no-store",
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}

// Proxies DELETE /v1/auth/me -- account deletion. Clears the session
// cookie on success (the backend already deleted every session row
// server-side, but the httpOnly cookie itself is this Next.js layer's
// responsibility, same as /api/auth/logout).
export async function DELETE(request: NextRequest) {
  const body = await request.json();
  const sessionToken = await getSessionToken();

  const resp = await fetch(`${FINSIGHT_API_URL}/v1/auth/me`, {
    method: "DELETE",
    headers: backendHeaders(sessionToken),
    body: JSON.stringify({ password: body.password }),
  });

  const data = await resp.json();
  if (resp.ok) {
    await clearSessionCookie();
  }
  return NextResponse.json(data, { status: resp.status });
}
