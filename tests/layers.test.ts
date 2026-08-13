import { describe, expect, it } from "vitest";
import { batchNorm, dense, dropout, embedding, layerNorm } from "../src/layers.js";

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

describe("batchNorm", () => {
  it("crea una spec de batchnorm", () => {
    expect(batchNorm(8)).toEqual({ kind: "batchnorm", features: 8 });
  });
});

describe("embedding", () => {
  it("crea una spec de embedding", () => {
    expect(embedding(100, 16)).toEqual({ kind: "embedding", vocab: 100, dim: 16 });
  });
});
