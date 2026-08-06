import type { Metadata, Viewport } from "next";
import { JetBrains_Mono } from "next/font/google";
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
    <html lang="en" className={`h-full antialiased ${jetbrainsMono.variable}`}>
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
