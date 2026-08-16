import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { proxyFetch } from "@/lib/proxyFetch";
import { setSessionCookie } from "@/lib/session";

// Proxies POST /v1/auth/login -- same session-cookie handling as
// signup/route.ts (see that file's comment).
export async function POST(request: NextRequest) {
  const body = await request.json();

  const resp = await proxyFetch(`${FINSIGHT_API_URL}/v1/auth/login`, {
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
