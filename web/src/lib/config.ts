// Server-side only -- never prefixed NEXT_PUBLIC_, so this never reaches
// the browser bundle. Every call to the backend API goes through this
// app's own /api/research/* route handlers (see src/app/api/research/),
// which run server-side and attach the key -- the browser only ever
// talks to this Next.js app, never directly to Cloud Run. Same
// credential-safety principle as streamlit_app.py, just enforced by
// Next.js's client/server split instead of Streamlit being all-server.
export const FINSIGHT_API_URL = (process.env.FINSIGHT_API_URL || "http://localhost:8000").replace(/\/$/, "");
export const FINSIGHT_API_KEY = process.env.FINSIGHT_API_KEY || "";

export function backendHeaders(extra?: Record<string, string>): HeadersInit {
  return {
    "Content-Type": "application/json",
    ...(FINSIGHT_API_KEY ? { "X-API-Key": FINSIGHT_API_KEY } : {}),
    ...extra,
  };
}
