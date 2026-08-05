// Real service worker (see web/src/lib/usePushNotifications.ts for
// registration) -- handles both Web Push display/click-to-focus AND
// offline caching (added after push notifications shipped -- this
// file's docstring used to explicitly scope offline out; that's the
// "last remaining feature" this update fills in).

// Version suffix, not just "finsight-cache" -- activate (below)
// deletes any cache NOT under this exact name, which is what makes a
// future change to this file (e.g. a different caching strategy) able
// to invalidate everything from a prior version instead of
// accumulating stale entries forever. Bump this string when the
// caching strategy itself changes, not on every unrelated edit.
const CACHE_NAME = "finsight-cache-v1";

self.addEventListener("install", () => {
  // Take over immediately rather than waiting for every open tab to
  // close first -- this is a small, low-risk hand-rolled worker, not
  // one where an in-flight update could corrupt anything mid-request.
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((name) => name !== CACHE_NAME).map((name) => caches.delete(name)))
    ).then(() => self.clients.claim())
  );
});

async function networkFirstWithShellFallback(request) {
  const cache = await caches.open(CACHE_NAME);
  try {
    const response = await fetch(request);
    // Only a genuinely successful response is worth caching as "the
    // app shell" -- caching an error page would make the offline
    // fallback show that error forever instead of the real page.
    if (response.ok) cache.put(request, response.clone());
    return response;
  } catch {
    // Offline, or the network request otherwise failed outright (not
    // the same as a slow-but-eventually-successful request, which
    // fetch() itself already waits out) -- fall back to whatever this
    // exact URL last resolved to, or failing that, the root shell,
    // since every route in this single-page app ultimately renders
    // the same page.
    const cached = (await cache.match(request)) || (await cache.match("/"));
    if (cached) return cached;
    // Nothing cached yet either (e.g. the very first navigation after
    // this worker activated, with an empty cache, during a genuine
    // network failure) -- event.respondWith() receiving undefined
    // here would hard-fail the whole navigation with no page at all,
    // not even a visible error. A synthesized minimal response is
    // strictly better than that.
    return new Response(
      "<!doctype html><title>Offline</title><p>You're offline and this page hasn't been cached yet. Reconnect and reload.</p>",
      { status: 503, headers: { "Content-Type": "text/html" } }
    );
  }
}

async function cacheFirstThenNetwork(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  if (response.ok) cache.put(request, response.clone());
  return response;
}

self.addEventListener("fetch", (event) => {
  const request = event.request;
  // Caches.put() throws for a non-GET request, and nothing here
  // should ever intercept a mutating call anyway -- everything below
  // this line only ever runs for GET.
  if (request.method !== "GET") return;

  const url = new URL(request.url);

  // Never cache backend calls, no exceptions -- /api/* covers every
  // Next.js proxy route (research submission/polling, auth, push),
  // all of which are dynamic and session-scoped. A cached job-status
  // poll that never updates, or a cached response served across
  // different sessions, is a real correctness bug, not a performance
  // win. Returning without calling event.respondWith() means this
  // request proceeds exactly as if there were no service worker at all.
  if (url.pathname.startsWith("/api/")) return;

  if (url.origin !== self.location.origin) return;

  if (request.mode === "navigate") {
    event.respondWith(networkFirstWithShellFallback(request));
    return;
  }

  // Everything else same-origin (Next.js's content-hashed /_next/static/*
  // chunks, icons, manifest.webmanifest) is safe to cache-first --
  // Next.js fingerprints these filenames per build, so the same URL
  // never resolves to different content later.
  event.respondWith(cacheFirstThenNetwork(request));
});

self.addEventListener("push", (event) => {
  // Payload shape matches app/api/auth.py's send_push_notification:
  // {"title": ..., "body": ..., "status": "done"|"error"}. Falls back
  // to generic text if the payload is missing/unparseable rather than
  // showing no notification at all -- a push event without a usable
  // body still means *something* finished.
  let payload = { title: "FinSight AI", body: "Your research report is ready." };
  if (event.data) {
    try {
      payload = event.data.json();
    } catch {
      // keep the default above
    }
  }

  event.waitUntil(
    self.registration.showNotification(payload.title, {
      body: payload.body,
      icon: "/icon-192.png",
      badge: "/icon-192.png",
      data: { url: "/" },
    })
  );
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();

  event.waitUntil(
    self.clients.matchAll({ type: "window", includeUncontrolled: true }).then((windowClients) => {
      for (const client of windowClients) {
        if (client.url.startsWith(self.location.origin) && "focus" in client) {
          return client.focus();
        }
      }
      if (self.clients.openWindow) {
        return self.clients.openWindow("/");
      }
    })
  );
});
