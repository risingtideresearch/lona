import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  root: fileURLToPath(new URL(".", import.meta.url)),
  resolve: {
    alias: {
      lona: fileURLToPath(
        new URL("../../packages/lona/src/main.ts", import.meta.url),
      ),
      "lona-geom": fileURLToPath(
        new URL("../../packages/lona-geom/src/main.ts", import.meta.url),
      ),
      "lona-curves": fileURLToPath(
        new URL("../../packages/lona-curves/src/main.ts", import.meta.url),
      ),
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
