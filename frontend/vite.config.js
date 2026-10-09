import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` serves the UI on http://localhost:5173 and forwards API calls to the FastAPI server on :7860.
// `npm run build` writes frontend/dist, which server.py serves at "/".
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      // Libraries in their own files, so a UI change does not make browsers download React and the charts again
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) return undefined;
          if (/node_modules[\\/](react|react-dom|scheduler)[\\/]/.test(id)) return "react";
          return "charts"; // recharts and its d3 / lodash dependencies
        },
      },
    },
  },
  server: {
    proxy: {
      "/v1": "http://localhost:7860",
      "/metrics": "http://localhost:7860",
    },
  },
});
