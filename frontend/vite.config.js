import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
const proxy = Object.fromEntries(
  [
    "/health",
    "/process",
    "/process-text",
    "/results",
    "/result/",
    "/live/start",
    "/live/stop",
    "/live/status",
    "/mic-test",
  ].map((route) => [
    route,
    {
      target: "http://127.0.0.1:8000",
      changeOrigin: true,
      timeout: 0,
      proxyTimeout: 0,
    },
  ]),
);
proxy["/live/ws"] = { target: "ws://127.0.0.1:8000", ws: true };
export default defineConfig({
  plugins: [
    react(),
    {
      name: "local-dev-csp",
      apply: "serve",
      transformIndexHtml(html) {
        return html.replace(
          "script-src 'self';",
          "script-src 'self' 'unsafe-inline';",
        );
      },
    },
  ],
  server: { host: "127.0.0.1", port: 5173, strictPort: true, proxy },
  build: { outDir: "dist" },
  preview: { port: 5173, strictPort: true, proxy },
});
