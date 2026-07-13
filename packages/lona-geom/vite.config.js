import { resolve } from "node:path";
import { createLibraryConfig } from "../../config/vite-library.js";

export default createLibraryConfig({
  packageDir: import.meta.dirname,
  name: "lona-geom",
  external: ["lona"],
  alias: [
    {
      find: /^lona$/,
      replacement: resolve(import.meta.dirname, "../lona/src/main.ts"),
    },
  ],
});
