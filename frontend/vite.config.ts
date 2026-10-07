import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The FastAPI backend (`council serve`, port 8000) serves the built `dist/` folder.
// During development, Vite serves the UI and forwards /api to the backend.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
  },
});
