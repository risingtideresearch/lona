import { resolve } from "node:path";
import { defineConfig } from "vite";
import dts from "vite-plugin-dts";

/**
 * Shared Vite configuration for Lona's publishable library packages.
 *
 * @param {{
 *   packageDir: string;
 *   name: string;
 *   external?: string[];
 *   alias?: import("vite").AliasOptions;
 * }} options
 */
export function createLibraryConfig({
  packageDir,
  name,
  external = [],
  alias,
}) {
  return defineConfig(({ mode }) => ({
    resolve: { alias },
    build: {
      minify: false,
      lib: {
        entry: resolve(packageDir, "src/main.ts"),
        name,
        fileName: name,
        formats: ["es"],
      },
      rollupOptions: { external },
    },
    plugins: [
      dts({
        entryRoot: "src",
        include: ["src"],
        exclude: [
          "src/**/*.test.ts",
          "src/**/*.integration.ts",
          "src/test-utils.ts",
        ],
        compilerOptions: {
          declarationMap: mode === "debug",
          declaration: mode === "debug",
        },
      }),
    ],
  }));
}
