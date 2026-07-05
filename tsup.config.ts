import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  outDir: "lib",
  format: ["esm", "cjs"],
  dts: true,
  clean: true,
  sourcemap: true,
  target: "node18",
  // Provee import.meta.url / __dirname en ambos formatos para el loader nativo.
  shims: true,
  // El binding nativo vive fuera del bundle; nunca intentar empaquetarlo.
  external: ["../index.cjs"],
});
