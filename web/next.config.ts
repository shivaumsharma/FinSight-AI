import type { NextConfig } from "next";

// Only for the render benchmark (bench/): lets React's Profiler report from a production build.
const nextConfig: NextConfig = {
  reactProductionProfiling: process.env.NEXT_PUBLIC_BENCH === "1",
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
        ],
      },
    ];
  },
};

export default nextConfig;
