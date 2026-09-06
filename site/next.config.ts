import type { NextConfig } from 'next';
const nextConfig:NextConfig={output:'export',basePath:process.env.GITHUB_PAGES==='1'?'/nfl-qb-attempts':'',images:{unoptimized:true}};
export default nextConfig;
