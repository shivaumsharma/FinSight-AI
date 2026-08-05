import { NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";

// Proxies GET /v1/push/vapid-public-key -- not a secret, no session
// needed (the backend route itself requires none either, see
// app/api/main.py's push_vapid_public_key).
export async function GET() {
  const resp = await fetch(`${FINSIGHT_API_URL}/v1/push/vapid-public-key`, {
    headers: backendHeaders(),
    cache: "no-store",
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
