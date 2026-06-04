import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Prevent bundling native Node.js modules
  serverExternalPackages: ['onnxruntime-node', 'sharp'],

  // Enable standalone output for Docker deployment
  output: 'standalone',

  // Image optimization settings
  images: {
    remotePatterns: [],
  },
};

export default nextConfig;
