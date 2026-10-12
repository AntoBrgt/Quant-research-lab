import type { NextConfig } from "next";

// Static export: the whole site is plain HTML/JS/CSS built from site/public/data/*.json.
// BASE_PATH is only set for the GitHub Pages fallback (served under /<repo>/).
const config: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
  basePath: process.env.BASE_PATH || undefined,
  poweredByHeader: false,
};

export default config;
