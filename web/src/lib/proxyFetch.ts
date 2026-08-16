// Every route under src/app/api/**/route.ts is a thin proxy to the
// FastAPI backend (FINSIGHT_API_URL, see config.ts) -- a bare fetch()
// to that backend throws when it's unreachable (down, cold-starting,
// timed out), and left uncaught that crashes the route handler, which
// Next.js turns into a generic, uninformative 500. Repeating a
// try/catch in all ~47 routes would be the same handful of lines
// copy-pasted everywhere, so it lives here once instead.
//
// Drop-in replacement for fetch() itself, not a new calling convention:
// every existing route already does
//   const resp = await fetch(url, init);
//   const data = await resp.json();
//   return NextResponse.json(data, { status: resp.status });
// and swapping in proxyFetch(url, init) keeps that shape working
// unchanged -- on a real network failure this returns a genuine
// Response (status BACKEND_UNREACHABLE_STATUS, valid JSON body) rather
// than throwing, so resp.ok/resp.status/resp.json() all still work.
//
// The response body's `code: "BACKEND_UNREACHABLE"` is what lets a
// caller (see useAuth.ts's checkSession) distinguish "the backend
// itself said no" from "we couldn't reach the backend at all" --
// those are different situations and must not be handled identically
// (see useAuth.ts's own comment on why /api/auth/me in particular
// cares about this).
export const BACKEND_UNREACHABLE_STATUS = 503;

export async function proxyFetch(input: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init);
  } catch {
    return new Response(
      JSON.stringify({
        code: "BACKEND_UNREACHABLE",
        message: "Couldn't reach the backend service. Please try again in a moment.",
      }),
      { status: BACKEND_UNREACHABLE_STATUS, headers: { "Content-Type": "application/json" } }
    );
  }
}
