import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { getSessionToken } from "@/lib/session";

// Proxies POST /v1/voice/synthesize -- JSON in, binary WAV out (the
// mirror image of pdf/route.ts's binary-passthrough pattern, but with
// a JSON request body like chat/route.ts rather than a GET).
export async function POST(request: NextRequest) {
  const body = await request.json();
  const sessionToken = await getSessionToken();

  const resp = await fetch(`${FINSIGHT_API_URL}/v1/voice/synthesize`, {
    method: "POST",
    headers: backendHeaders(sessionToken),
    body: JSON.stringify({ text: body.text }),
  });

  if (!resp.ok) {
    const data = await resp.json().catch(() => ({ code: "TTS_FETCH_FAILED", message: resp.statusText }));
    return NextResponse.json(data, { status: resp.status });
  }

  const bytes = await resp.arrayBuffer();
  return new NextResponse(bytes, {
    status: 200,
    headers: { "Content-Type": "audio/wav" },
  });
}
