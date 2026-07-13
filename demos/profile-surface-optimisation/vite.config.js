import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";

export default defineConfig({
  root: fileURLToPath(new URL(".", import.meta.url)),
  base: "./",
  server: {
    port: 5174,
    open: false,
  },
  resolve: {
    alias: {
      lona: fileURLToPath(
        new URL("../../packages/lona/src/main.ts", import.meta.url),
      ),
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
