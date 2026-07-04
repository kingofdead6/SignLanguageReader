// vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// FastAPI backend (server.py). 127.0.0.1 on purpose — on Windows,
// "localhost" can resolve to IPv6 ::1 while uvicorn listens on IPv4,
// which makes the proxy throw ECONNREFUSED.
const BACKEND_HTTP = "http://127.0.0.1:8000";
const BACKEND_WS = "ws://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()], // keep YOUR existing plugins line if different
  server: {
    proxy: {
      "/health":  { target: BACKEND_HTTP, changeOrigin: true },
      "/poses":   { target: BACKEND_HTTP, changeOrigin: true },
      "/predict": { target: BACKEND_HTTP, changeOrigin: true },
      "/sign":    { target: BACKEND_HTTP, changeOrigin: true },
      "/ws":      { target: BACKEND_WS, ws: true, changeOrigin: true },
    },
  },
});