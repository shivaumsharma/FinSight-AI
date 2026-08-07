import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { getSessionToken } from "@/lib/session";

// Proxies POST /v1/research/{job_id}/model-compare.
export async function POST(
  request: NextRequest,
  context: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await context.params;
  const sessionToken = await getSessionToken();

  const resp = await fetch(`${FINSIGHT_API_URL}/v1/research/${jobId}/model-compare`, {
    method: "POST",
    headers: backendHeaders(sessionToken),
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
