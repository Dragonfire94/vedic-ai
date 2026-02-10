/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000',
  },
  images: {
    domains: ['vedic-ai-production.up.railway.app'],
  },
}

module.exports = nextConfig
