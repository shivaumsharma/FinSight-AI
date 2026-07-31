import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";

// Proxies GET /v1/research/{job_id} -- polled by the client every few
// seconds while a job is running/queued.
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await context.params;

  const resp = await fetch(`${FINSIGHT_API_URL}/v1/research/${jobId}`, {
    headers: backendHeaders(),
    cache: "no-store",
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
