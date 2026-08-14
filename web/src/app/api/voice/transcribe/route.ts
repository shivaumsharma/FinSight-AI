import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, FINSIGHT_API_KEY } from "@/lib/config";
import { proxyFetch } from "@/lib/proxyFetch";
import { getSessionToken } from "@/lib/session";

// Proxies POST /v1/voice/transcribe -- the FIRST multipart proxy route
// in this app (every other route under src/app/api/* forwards JSON via
// backendHeaders(), see e.g. research/route.ts). Re-parses the incoming
// request as FormData and rebuilds a fresh FormData to forward, rather
// than streaming the raw body through -- this lets fetch() generate its
// own correct `multipart/form-data; boundary=...` header for the
// rebuilt body. Deliberately does NOT set Content-Type manually: the
// original request's boundary string is only valid for the original
// body, not this rebuilt one -- setting it by hand here would produce a
// header/body mismatch the backend can't parse.
export async function POST(request: NextRequest) {
  const incoming = await request.formData();
  const file = incoming.get("file");
  if (!(file instanceof Blob)) {
    return NextResponse.json({ code: "INVALID_AUDIO", message: "No audio file provided." }, { status: 400 });
  }

  const sessionToken = await getSessionToken();
  const outgoing = new FormData();
  outgoing.set("file", file, "recording.wav");

  const resp = await proxyFetch(`${FINSIGHT_API_URL}/v1/voice/transcribe`, {
    method: "POST",
    headers: {
      ...(FINSIGHT_API_KEY ? { "X-API-Key": FINSIGHT_API_KEY } : {}),
      ...(sessionToken ? { Authorization: `Bearer ${sessionToken}` } : {}),
    },
    body: outgoing,
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
