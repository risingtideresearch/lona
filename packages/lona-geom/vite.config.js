/* global __dirname */
import { resolve } from "path";
import { defineConfig } from "vite";
import dts from "vite-plugin-dts";

export default defineConfig({
  build: {
    minify: false,
    lib: {
      entry: resolve(__dirname, "src/main.ts"),
      name: "lona-geom",
      fileName: "lona-geom",
      formats: ["es"],
    },
    rollupOptions: {
      external: ["lona"],
    },
  },
  plugins: [dts()],
});
