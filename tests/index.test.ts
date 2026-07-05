import { describe, expect, it } from "vitest";
import intellivium, { Model, Tensor, dense, tensor } from "../src/index.js";

describe("public API", () => {
  it("exporta las funciones y clases principales", () => {
    expect(typeof tensor).toBe("function");
    expect(typeof dense).toBe("function");
    expect(typeof Model).toBe("function");
    expect(typeof Tensor).toBe("function");
  });

  it("el export por defecto agrupa todo", () => {
    expect(intellivium.tensor).toBe(tensor);
    expect(intellivium.dense).toBe(dense);
    expect(intellivium.Model).toBe(Model);
    expect(intellivium.Tensor).toBe(Tensor);
  });
});
