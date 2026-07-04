// Central API configuration.
//
// DEV (default, VITE_API_BASE unset):
//   - HTTP -> same-origin /poses, /sign, ... proxied by Vite (vite.config.js)
//   - WS   -> DIRECT to the backend. WebSockets aren't subject to CORS,
//     so the proxy adds nothing — and Vite's ws proxy is flaky on
//     Windows (write ECONNABORTED), so we bypass it.
//
// PROD: set VITE_API_BASE=https://your-api-domain.com in .env before build.
const raw = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

export const API_BASE = raw
  ? raw.replace("//0.0.0.0", "//localhost")
  : ""; // empty, NOT "/" — "//poses" would be a protocol-relative URL

export const WS_URLS = (() => {
  if (!raw) {
    // Direct to the backend in dev; IPv4 loopback first, localhost fallback.
    return ["ws://127.0.0.1:8000/ws", "ws://localhost:8000/ws"];
  }
  const toWs = (b) => b.replace(/^http/, "ws") + "/ws";
  const urls = [toWs(API_BASE)];
  if (API_BASE.includes("//localhost")) {
    urls.push(toWs(API_BASE.replace("//localhost", "//127.0.0.1")));
  } else if (API_BASE.includes("//127.0.0.1")) {
    urls.push(toWs(API_BASE.replace("//127.0.0.1", "//localhost")));
  }
  return urls;
})();