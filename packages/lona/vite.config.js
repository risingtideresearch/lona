/* global __dirname */
import { resolve } from "path";
import { defineConfig } from "vite";
import dts from "vite-plugin-dts";

export default defineConfig(({ mode }) => {
  return {
    build: {
      minify: false,
      lib: {
        // Could also be a dictionary or array of multiple entry points
        entry: resolve(__dirname, "src/main.ts"),
        name: "lona",
        // the proper extensions will be added
        fileName: "lona",
        formats: ["es"],
      },
      rollupOptions: {
        external: ["webgpu"],
      },
    },
    plugins: [
      dts({
        entryRoot: "src",
        compilerOptions: {
          declarationMap: mode === "debug",
          declaration: mode === "debug",
        },
      }),
    ],
  };
});
