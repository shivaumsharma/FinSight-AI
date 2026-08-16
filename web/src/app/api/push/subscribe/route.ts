import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { proxyFetch } from "@/lib/proxyFetch";
import { getSessionToken } from "@/lib/session";

// Proxies POST /v1/push/subscribe. Body is the browser's own
// PushSubscription.toJSON() output ({endpoint, keys: {p256dh, auth}})
// forwarded unmodified -- see usePushNotifications.ts's subscribe().
export async function POST(request: NextRequest) {
  const body = await request.json();
  const sessionToken = await getSessionToken();

  const resp = await proxyFetch(`${FINSIGHT_API_URL}/v1/push/subscribe`, {
    method: "POST",
    headers: backendHeaders(sessionToken),
    body: JSON.stringify(body),
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
