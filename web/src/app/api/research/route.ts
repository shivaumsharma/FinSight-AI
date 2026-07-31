import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";

// Proxies POST /v1/research on the FastAPI backend. Runs server-side
// so FINSIGHT_API_KEY (backendHeaders) never reaches the browser.
export async function POST(request: NextRequest) {
  const body = await request.json();

  const resp = await fetch(`${FINSIGHT_API_URL}/v1/research`, {
    method: "POST",
    headers: backendHeaders(),
    body: JSON.stringify({ question: body.question }),
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
