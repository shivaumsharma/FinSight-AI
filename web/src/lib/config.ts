// Server-side only -- never prefixed NEXT_PUBLIC_, so this never reaches
// the browser bundle. Every call to the backend API goes through this
// app's own /api/research/* route handlers (see src/app/api/research/),
// which run server-side and attach the key -- the browser only ever
// talks to this Next.js app, never directly to Cloud Run. Same
// credential-safety principle as streamlit_app.py, just enforced by
// Next.js's client/server split instead of Streamlit being all-server.
export const FINSIGHT_API_URL = (process.env.FINSIGHT_API_URL || "http://localhost:8000").replace(/\/$/, "");
export const FINSIGHT_API_KEY = process.env.FINSIGHT_API_KEY || "";

// sessionToken carries the per-user identity (see src/lib/session.ts) --
// FINSIGHT_API_KEY alone only proves "this deployment", not "this
// user" (see app/api/main.py's own comment on why both auth layers
// exist). Optional so routes that genuinely don't need a logged-in
// user (there are none among /v1/research/* anymore, but a future
// public route might) aren't forced to pass one.
export function backendHeaders(sessionToken?: string, extra?: Record<string, string>): HeadersInit {
  return {
    "Content-Type": "application/json",
    ...(FINSIGHT_API_KEY ? { "X-API-Key": FINSIGHT_API_KEY } : {}),
    ...(sessionToken ? { Authorization: `Bearer ${sessionToken}` } : {}),
    ...extra,
  };
}
