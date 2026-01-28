import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        // Supabase Storage - uses wildcard for any project subdomain
        protocol: 'https',
        hostname: '**.supabase.co',
        pathname: '/storage/v1/object/public/**',
      },
      {
        // Render backend service
        protocol: 'https',
        hostname: '**.onrender.com',
        pathname: '/**',
      },
      {
        // LlamaParse images (temporary URLs)
        protocol: 'https',
        hostname: '**.amazonaws.com',
        pathname: '/**',
      },
      {
        // Local development
        protocol: 'http',
        hostname: 'localhost',
        port: '8000',
        pathname: '/**',
      },
    ],
    // Allow unoptimized images for external sources (required for dynamic URLs)
    unoptimized: true,
  },
  // API rewrites for development (Vercel handles this via env vars in production)
  async rewrites() {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    return [
      {
        source: '/backend/:path*',
        destination: `${apiUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
