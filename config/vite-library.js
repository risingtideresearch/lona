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
 *   entries?: Record<string, string>;
 *   dtsBeforeWriteFile?: (filePath: string, content: string) => void | false | { filePath?: string; content?: string };
 * }} options
 */
export function createLibraryConfig({
  packageDir,
  name,
  external = [],
  alias,
  entries,
  dtsBeforeWriteFile,
}) {
  return defineConfig(({ mode }) => ({
    resolve: { alias },
    build: {
      minify: false,
      lib: {
        entry: entries
          ? Object.fromEntries(
              Object.entries(entries).map(([entryName, path]) => [
                entryName,
                resolve(packageDir, path),
              ]),
            )
          : resolve(packageDir, "src/main.ts"),
        name,
        fileName: entries ? (_format, entryName) => `${entryName}.js` : name,
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
        beforeWriteFile: dtsBeforeWriteFile,
      }),
    ],
  }));
}
