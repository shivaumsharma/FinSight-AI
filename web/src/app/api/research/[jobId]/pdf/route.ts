import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { getSessionToken } from "@/lib/session";

// Proxies GET /v1/research/{job_id}/pdf -- binary passthrough, not JSON.
// A visitor opening a shared link (see .../pdf/share/route.ts) has no
// session cookie at all -- exp/sig are forwarded unchanged so the
// backend's own signature check (app/api/auth.py's verify_pdf_signature)
// can grant access independent of the session-token path below.
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await context.params;
  const sessionToken = await getSessionToken();

  const backendUrl = new URL(`${FINSIGHT_API_URL}/v1/research/${jobId}/pdf`);
  const exp = request.nextUrl.searchParams.get("exp");
  const sig = request.nextUrl.searchParams.get("sig");
  if (exp) backendUrl.searchParams.set("exp", exp);
  if (sig) backendUrl.searchParams.set("sig", sig);

  const resp = await fetch(backendUrl.toString(), {
    headers: backendHeaders(sessionToken),
    cache: "no-store",
  });

  if (!resp.ok) {
    const data = await resp.json().catch(() => ({ code: "PDF_FETCH_FAILED", message: resp.statusText }));
    return NextResponse.json(data, { status: resp.status });
  }

  const bytes = await resp.arrayBuffer();
  return new NextResponse(bytes, {
    status: 200,
    headers: {
      "Content-Type": "application/pdf",
      "Content-Disposition": `attachment; filename="${jobId}.pdf"`,
    },
  });
}
