"use client";

import { useEffect } from "react";

// Registers /sw.js unconditionally on every page load -- previously
// this only happened inside usePushNotifications.ts, which only runs
// on the Profile page and NotificationsPanel. An active, controlling
// service worker is one of Chrome's own installability criteria for
// the "Add to Home Screen" prompt (alongside the manifest and HTTPS),
// so a user who never opened Profile never had a working service
// worker and the app was never actually installable for them, despite
// manifest.ts/sw.js/the icons all being fully built already. Mounted
// once in the root layout so it runs regardless of which page a
// session starts on. Registration itself is idempotent -- calling it
// again from usePushNotifications.ts later just returns the same
// registration, no conflict.
export default function ServiceWorkerRegistration() {
  useEffect(() => {
    if (!("serviceWorker" in navigator)) return;
    navigator.serviceWorker.register("/sw.js").catch(() => {
      // Best-effort: a failed registration here just means no offline
      // caching/install prompt this session, not a broken app --
      // every actual page request already works without it.
    });
  }, []);

  return null;
}
