import { createLibraryConfig } from "../../config/vite-library.js";

// The workspace tsconfig maps the bare specifier "lona" to that package's
// SOURCE (`../lona/src/main.ts`), for fast in-monorepo dev type-checking.
// vite-plugin-dts bakes that resolved source path into the emitted .d.ts,
// which breaks type identity for external consumers: a downstream package
// importing both "lona" (resolving to lona's published dist) and
// "lona-columnar" (whose .d.ts pointed at lona's *source*) sees two
// nominally distinct `Num` classes — they disagree on private fields — so
// anything that structurally depends on `Num`/`NumStruct` (this package's
// whole public surface) fails to type-check downstream. Rewrite the
// resolved import back to the bare "lona"/"lona/internal" specifiers so
// every consumer, in or out of this workspace, resolves the same one.
const LONA_SOURCE_IMPORT =
  /from ["']([^"']*\/lona\/src\/(main|internal)\.ts)["']/g;

export default createLibraryConfig({
  packageDir: import.meta.dirname,
  name: "lona-columnar",
  external: ["lona", "lona/internal"],
  dtsBeforeWriteFile: (_filePath, content) => {
    if (!LONA_SOURCE_IMPORT.test(content)) return;
    LONA_SOURCE_IMPORT.lastIndex = 0;
    return {
      content: content.replace(
        LONA_SOURCE_IMPORT,
        (_match, _path, entry) =>
          `from "lona${entry === "internal" ? "/internal" : ""}"`,
      ),
    };
  },
});
