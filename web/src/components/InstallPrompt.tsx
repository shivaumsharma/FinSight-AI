"use client";

import { useEffect, useState } from "react";

// Chrome/Android show their own install prompt automatically once the
// manifest + HTTPS criteria are met -- no UI needed here for them. iOS
// Safari never fires that prompt at all (no beforeinstallprompt event),
// so this only renders the "how to install" instructions on iOS, and
// only if not already running installed (display-mode: standalone).
//
// navigator/window are only defined client-side -- this state can't be
// computed during a lazy useState initializer (that would crash on the
// server-rendered pass of this client component) or read directly in
// the render body, so it's read once in an effect and applied as a
// single combined update (not two separate setState calls, which
// would otherwise trigger a second cascading render).
export default function InstallPrompt() {
  const [state, setState] = useState({ isIOS: false, isStandalone: true });

  useEffect(() => {
    // Reading navigator/window once on mount to detect iOS + standalone
    // display mode; this is Next.js's own documented pattern for this
    // exact PWA install-prompt case (see their progressive-web-apps.md
    // guide). No render-body alternative exists: these globals aren't
    // defined during this client component's server-rendered pass.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setState({
      isIOS: /iPad|iPhone|iPod/.test(navigator.userAgent),
      isStandalone: window.matchMedia("(display-mode: standalone)").matches,
    });
  }, []);

  if (!state.isIOS || state.isStandalone) return null;

  return (
    <div className="mb-4 rounded-lg border border-border bg-card px-3.5 py-2.5 text-xs text-muted">
      Install this app: tap <span className="font-semibold text-text">Share</span> then{" "}
      <span className="font-semibold text-text">Add to Home Screen</span>.
    </div>
  );
}
