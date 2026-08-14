import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  basePath: "/muon-norm-cap-grokking",
  assetPrefix: "/muon-norm-cap-grokking",
  images: { unoptimized: true },
};

export default nextConfig;
