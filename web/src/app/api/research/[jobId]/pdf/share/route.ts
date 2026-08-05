import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { getSessionToken } from "@/lib/session";

// Proxies POST /v1/research/{job_id}/pdf/share. The backend returns a
// relative /v1/research/... path+query -- reshaped here into an
// absolute /api/research/... URL on THIS app's own origin, since the
// browser never talks to the FastAPI backend directly (see config.ts)
// and a link a user actually shares needs to be one their recipient's
// browser can hit without knowing FINSIGHT_API_URL exists at all.
export async function POST(
  request: NextRequest,
  context: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await context.params;
  const sessionToken = await getSessionToken();

  const resp = await fetch(`${FINSIGHT_API_URL}/v1/research/${jobId}/pdf/share`, {
    method: "POST",
    headers: backendHeaders(sessionToken),
  });

  const data = await resp.json();
  if (!resp.ok) {
    return NextResponse.json(data, { status: resp.status });
  }

  const backendPath: string = data.url; // "/v1/research/{job_id}/pdf?exp=...&sig=..."
  const query = backendPath.split("?", 2)[1] ?? "";
  const shareUrl = `${request.nextUrl.origin}/api/research/${jobId}/pdf${query ? `?${query}` : ""}`;

  return NextResponse.json({ url: shareUrl, expires_at: data.expires_at });
}
