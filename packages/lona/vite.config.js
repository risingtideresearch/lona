import { createLibraryConfig } from "../../config/vite-library.js";

export default createLibraryConfig({
  packageDir: import.meta.dirname,
  name: "lona",
  entries: {
    lona: "src/main.ts",
    internal: "src/internal.ts",
  },
  external: ["webgpu"],
});
