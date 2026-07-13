/**
 * Register every backend by importing each subdirectory's index for its
 * side effects. Each backend registers both its base and -sym variants.
 */
import "./js-interp";
import "./js-codegen";
import "./wasm-interp";
import "./wasm-codegen";
import "./gpu-interp";
import "./gpu-codegen";
