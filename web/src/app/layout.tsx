import type { Metadata, Viewport } from "next";
import { JetBrains_Mono, Source_Serif_4 } from "next/font/google";
import Script from "next/script";
import "./globals.css";

// The whole UI is built on a monospace terminal aesthetic (see
// globals.css), but the old --font-mono stack (ui-monospace, Menlo,
// SF Mono) is Mac-only -- Windows Chrome has none of those and falls
// back to plain "monospace", which resolves to Courier New. Loading a
// real webfont here makes the intended look render consistently
// everywhere instead of only on Mac.
const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono-loaded",
  display: "swap",
});

// Light mode's "Classical" look (see globals.css's [data-theme="light"]
// block) swaps --font-mono-loaded to this serif at the CSS-variable
// level instead of touching every component's `font-mono` className --
// every element that already renders in the dark theme's monospace
// automatically renders in this serif under light mode, no per-
// component changes needed.
const sourceSerif = Source_Serif_4({
  subsets: ["latin"],
  variable: "--font-serif-loaded",
  display: "swap",
});

// Runs before hydration (see the inline <script> in <head> below) --
// reads the persisted theme choice and stamps it onto <html> before
// first paint, so a light-mode user never sees a flash of the dark
// theme on load. Kept as a plain string, not an imported module: it
// has to execute synchronously and inline, before any bundled JS runs.
const THEME_INIT_SCRIPT = `(function(){try{if(localStorage.getItem('finsight:theme')==='light'){document.documentElement.setAttribute('data-theme','light');}}catch(e){}})();`;

export const metadata: Metadata = {
  title: "Finsight AI",
  description: "Autonomous Financial Intelligence Platform",
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "Finsight AI",
  },
  icons: {
    icon: [{ url: "/icon-512.png", sizes: "512x512", type: "image/png" }],
    apple: [{ url: "/apple-icon.png", sizes: "180x180", type: "image/png" }],
  },
};

export const viewport: Viewport = {
  themeColor: "#05070a",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`h-full antialiased ${jetbrainsMono.variable} ${sourceSerif.variable}`}>
      <body className="min-h-full flex flex-col">{children}</body>
      <Script id="theme-init" strategy="beforeInteractive">
        {THEME_INIT_SCRIPT}
      </Script>
    </html>
  );
}
