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
      // Use the local package source in dev/build, so the demo follows changes
      // to lona without requiring a library rebuild.
      lona: fileURLToPath(new URL("../../src/main.ts", import.meta.url)),
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
