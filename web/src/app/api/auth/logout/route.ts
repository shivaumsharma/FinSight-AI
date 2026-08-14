import { NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { proxyFetch } from "@/lib/proxyFetch";
import { clearSessionCookie, getSessionToken } from "@/lib/session";

// Proxies POST /v1/auth/logout. Clears the local cookie unconditionally,
// even if the backend call fails (a network blip shouldn't leave the
// user stuck "logged in" client-side with a token the backend may or
// may not still honor) -- the backend call is best-effort cleanup of
// that one session row, not a precondition for the browser forgetting it.
export async function POST() {
  const sessionToken = await getSessionToken();

  if (sessionToken) {
    await proxyFetch(`${FINSIGHT_API_URL}/v1/auth/logout`, {
      method: "POST",
      headers: backendHeaders(sessionToken),
    }).catch(() => {
      // Best-effort -- see this function's own comment above.
    });
  }

  await clearSessionCookie();
  return NextResponse.json({ status: "ok" });
}
