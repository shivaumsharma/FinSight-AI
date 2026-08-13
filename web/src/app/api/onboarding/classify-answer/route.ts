import { NextRequest, NextResponse } from "next/server";
import { FINSIGHT_API_URL, backendHeaders } from "@/lib/config";
import { getSessionToken } from "@/lib/session";

// Proxies POST /v1/onboarding/classify-answer.
export async function POST(request: NextRequest) {
  const body = await request.json();
  const sessionToken = await getSessionToken();

  const resp = await fetch(`${FINSIGHT_API_URL}/v1/onboarding/classify-answer`, {
    method: "POST",
    headers: backendHeaders(sessionToken),
    body: JSON.stringify({ field: body.field, answer: body.answer }),
  });

  const data = await resp.json();
  return NextResponse.json(data, { status: resp.status });
}
