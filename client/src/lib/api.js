// Central API configuration.
//
// DEV (VITE_API_BASE unset):
//   HTTP -> same-origin, proxied by Vite; WS -> direct to local backend.
// PROD:
//   Set VITE_API_BASE in .env.production, then `npm run build`.
//   (To test the hosted API from the dev server: put it in .env.local
//   and restart `npm run dev`.)
const raw = (import.meta.env.VITE_API_BASE || "").replace(/\/$/, "");

export const API_BASE = raw
  ? raw.replace("//0.0.0.0", "//localhost")
  : ""; // empty, NOT "/" — "//poses" would be a protocol-relative URL

export const WS_URLS = (() => {
  if (!raw) {
    return ["ws://127.0.0.1:8000/ws", "ws://localhost:8000/ws"];
  }
  const toWs = (b) => b.replace(/^http/, "ws") + "/ws"; // https -> wss
  const urls = [toWs(API_BASE)];
  if (API_BASE.includes("//localhost")) {
    urls.push(toWs(API_BASE.replace("//localhost", "//127.0.0.1")));
  } else if (API_BASE.includes("//127.0.0.1")) {
    urls.push(toWs(API_BASE.replace("//127.0.0.1", "//localhost")));
  }
  return urls;
})();