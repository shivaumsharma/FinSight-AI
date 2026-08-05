import { NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { getSessionToken } from "@/lib/session";

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
