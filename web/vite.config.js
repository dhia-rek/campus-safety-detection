import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The FastAPI backend (src/dashboard/api.py) runs on :8000.
// Proxy /api there so the front-end can use same-origin relative URLs.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
});
