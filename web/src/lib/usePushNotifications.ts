"use client";

import { useCallback, useEffect, useState } from "react";

type PushStatus = "checking" | "unsupported" | "unsubscribed" | "subscribed" | "denied";

// PushManager.subscribe()'s applicationServerKey wants a raw
// BufferSource, not the base64url string the backend hands back --
// standard conversion, no library needed for something this small.
// Returns BufferSource, not Uint8Array, deliberately -- newer
// TypeScript's generic Uint8Array<ArrayBufferLike> isn't directly
// assignable to applicationServerKey's DOM-lib type (ArrayBufferLike
// includes SharedArrayBuffer, which BufferSource's strict typing
// excludes), even though the actual runtime value here is always a
// plain, freshly-allocated ArrayBuffer -- a type-level mismatch, not a
// real one.
function urlBase64ToUint8Array(base64: string): BufferSource {
  const padding = "=".repeat((4 - (base64.length % 4)) % 4);
  const b64 = (base64 + padding).replace(/-/g, "+").replace(/_/g, "/");
  const raw = atob(b64);
  return Uint8Array.from([...raw].map((c) => c.charCodeAt(0))) as BufferSource;
}

// Mirrors useAuth.ts/useResearch.ts's shape -- status state plus
// imperative actions, same pattern as every other client/server
// round-trip in this app.
export function usePushNotifications() {
  const [status, setStatus] = useState<PushStatus>("checking");

  const checkStatus = useCallback(async () => {
    if (!("serviceWorker" in navigator) || !("PushManager" in window)) {
      setStatus("unsupported");
      return;
    }
    if (Notification.permission === "denied") {
      setStatus("denied");
      return;
    }
    try {
      const registration = await navigator.serviceWorker.register("/sw.js");
      const existing = await registration.pushManager.getSubscription();
      setStatus(existing ? "subscribed" : "unsubscribed");
    } catch {
      setStatus("unsupported");
    }
  }, []);

  useEffect(() => {
    checkStatus();
  }, [checkStatus]);

  const subscribe = useCallback(async () => {
    // Explicit user-initiated call only (see AuthGate/page.tsx's own
    // button) -- requestPermission() must never fire automatically on
    // mount, since an unsolicited prompt is exactly what gets a site's
    // notifications permanently blocked by the browser.
    const permission = await Notification.requestPermission();
    if (permission !== "granted") {
      setStatus("denied");
      return false;
    }

    const registration = await navigator.serviceWorker.ready;
    const keyResp = await fetch("/api/push/vapid-public-key");
    const { public_key } = await keyResp.json();

    const subscription = await registration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: urlBase64ToUint8Array(public_key),
    });

    await fetch("/api/push/subscribe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(subscription.toJSON()),
    });

    setStatus("subscribed");
    return true;
  }, []);

  const unsubscribe = useCallback(async () => {
    const registration = await navigator.serviceWorker.ready;
    const subscription = await registration.pushManager.getSubscription();
    if (subscription) {
      await fetch("/api/push/unsubscribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ endpoint: subscription.endpoint }),
      });
      await subscription.unsubscribe();
    }
    setStatus("unsubscribed");
  }, []);

  return { status, subscribe, unsubscribe };
}
