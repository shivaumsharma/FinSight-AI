import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";

// Proxies GET /v1/research/{job_id}/pdf -- binary passthrough, not JSON.
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await context.params;

  const resp = await fetch(`${FINSIGHT_API_URL}/v1/research/${jobId}/pdf`, {
    headers: backendHeaders(),
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
