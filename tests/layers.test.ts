import { describe, expect, it } from "vitest";
import { dense } from "../src/layers.js";

describe("dense", () => {
  it("crea una spec con la activación dada", () => {
    expect(dense(4, 8, "relu")).toEqual({
      inputDim: 4,
      outputDim: 8,
      activation: "relu",
    });
  });

  it("usa 'linear' por defecto", () => {
    expect(dense(2, 1)).toEqual({ inputDim: 2, outputDim: 1, activation: "linear" });
  });
});
