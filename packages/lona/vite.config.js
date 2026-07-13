import { createLibraryConfig } from "../../config/vite-library.js";

export default createLibraryConfig({
  packageDir: import.meta.dirname,
  name: "lona",
  external: ["webgpu"],
});
