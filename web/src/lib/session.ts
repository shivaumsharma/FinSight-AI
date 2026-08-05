import { cookies } from "next/headers";

// httpOnly session cookie holding the FastAPI-issued session token --
// never readable from client-side JS, same credential-safety principle
// as FINSIGHT_API_KEY in config.ts (that one's server-env-only, this
// one's httpOnly-cookie-only; neither ever reaches browser-executable
// code). 30 days matches app/api/db.py's DEFAULT_SESSION_TTL_SECONDS --
// keep these in sync if either changes.
export const SESSION_COOKIE_NAME = "finsight_session";
const SESSION_MAX_AGE_SECONDS = 30 * 24 * 3600;

// cookies() is async in this Next.js version (see cookies.md's own
// version history -- synchronous access was deprecated starting
// v15.0.0-RC) -- every caller must await this.
export async function getSessionToken(): Promise<string | undefined> {
  const store = await cookies();
  return store.get(SESSION_COOKIE_NAME)?.value;
}

export async function setSessionCookie(token: string): Promise<void> {
  const store = await cookies();
  store.set(SESSION_COOKIE_NAME, token, {
    httpOnly: true,
    // secure:true cookies are silently dropped over plain http --
    // would break session cookies entirely on local dev (http://localhost),
    // so this only turns on for a real production (https) deployment.
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: SESSION_MAX_AGE_SECONDS,
  });
}

export async function clearSessionCookie(): Promise<void> {
  const store = await cookies();
  store.delete(SESSION_COOKIE_NAME);
}
