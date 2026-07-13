import { resolve } from "node:path";
import { createLibraryConfig } from "../../config/vite-library.js";

export default createLibraryConfig({
  packageDir: import.meta.dirname,
  name: "lona-curves",
  external: ["lona", "lona-geom"],
  alias: [
    {
      find: /^lona$/,
      replacement: resolve(import.meta.dirname, "../lona/src/main.ts"),
    },
    {
      find: /^lona-geom$/,
      replacement: resolve(import.meta.dirname, "../lona-geom/src/main.ts"),
    },
  ],
});
