import { describe, expect, it } from "vitest";
import { dense, dropout, layerNorm } from "../src/layers.js";

describe("dense", () => {
  it("crea una spec con la activación dada", () => {
    expect(dense(4, 8, "relu")).toEqual({
      kind: "dense",
      inputDim: 4,
      outputDim: 8,
      activation: "relu",
    });
  });

  it("usa 'linear' por defecto", () => {
    expect(dense(2, 1)).toEqual({
      kind: "dense",
      inputDim: 2,
      outputDim: 1,
      activation: "linear",
    });
  });
});

describe("dropout", () => {
  it("crea una spec de dropout", () => {
    expect(dropout(0.3)).toEqual({ kind: "dropout", p: 0.3 });
  });
});

describe("layerNorm", () => {
  it("crea una spec de layernorm", () => {
    expect(layerNorm(16)).toEqual({ kind: "layernorm", features: 16 });
  });
});
