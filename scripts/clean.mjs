import { rm } from "node:fs/promises";
import { resolve } from "node:path";

const paths = process.argv.slice(2);
if (paths.length === 0) {
  throw new Error("Usage: node scripts/clean.mjs <path> [...paths]");
}

await Promise.all(
  paths.map((path) => rm(resolve(path), { recursive: true, force: true })),
);
