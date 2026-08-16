import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { proxyFetch } from "@/lib/proxyFetch";
import { setSessionCookie } from "@/lib/session";

// Proxies POST /v1/auth/signup. On success, stores the returned
// session_token as an httpOnly cookie server-side -- the browser never
// sees the raw token, same principle as FINSIGHT_API_KEY never
// reaching the browser bundle (see config.ts).
export async function POST(request: NextRequest) {
  const body = await request.json();

  const resp = await proxyFetch(`${FINSIGHT_API_URL}/v1/auth/signup`, {
    method: "POST",
    headers: backendHeaders(),
    body: JSON.stringify({ email: body.email, password: body.password }),
  });

  const data = await resp.json();

  if (resp.ok && data.session_token) {
    await setSessionCookie(data.session_token);
    return NextResponse.json({ status: "ok" }, { status: 200 });
  }

  return NextResponse.json(data, { status: resp.status });
}
