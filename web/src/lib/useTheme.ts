"use client";

import { useCallback, useEffect, useState } from "react";

export type Theme = "dark" | "light";

const STORAGE_KEY = "finsight:theme";

// Dark is the app's original, still-default look (see globals.css) --
// light is opt-in only, never assumed from the browser/OS's own
// color-scheme preference, so a user's device switching to light mode
// doesn't silently change how the app looks without them having
// chosen it here. layout.tsx's inline beforeInteractive script already
// applies the stored choice to <html> before first paint (avoiding a
// flash of the wrong theme); this hook just mirrors that same value
// into React state for whatever renders the toggle.
function readStoredTheme(): Theme {
  if (typeof window === "undefined") return "dark";
  return window.localStorage.getItem(STORAGE_KEY) === "light" ? "light" : "dark";
}

export function useTheme() {
  const [theme, setThemeState] = useState<Theme>("dark");

  useEffect(() => {
    setThemeState(readStoredTheme());
  }, []);

  const setTheme = useCallback((next: Theme) => {
    setThemeState(next);
    window.localStorage.setItem(STORAGE_KEY, next);
    if (next === "light") {
      document.documentElement.setAttribute("data-theme", "light");
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
  }, []);

  return { theme, setTheme };
}
