import { describe, expect, it } from "vitest";
import { Tensor, tensor } from "../src/tensor.js";

describe("Tensor", () => {
  it("construye desde un arreglo 2D con shape correcto", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(t.shape).toEqual([2, 3]);
    expect(t.rows).toBe(2);
    expect(t.cols).toBe(3);
  });

  it("almacena los datos en orden row-major", () => {
    const t = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(Array.from(t.data)).toEqual([1, 2, 3, 4]);
  });

  it("hace round-trip con toArray()", () => {
    const arr = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ];
    expect(tensor(arr).toArray()).toEqual(arr);
  });

  it("lanza error si las filas no son rectangulares", () => {
    expect(() => tensor([[1, 2], [3]])).toThrow();
  });

  it("maneja un tensor vacío", () => {
    const t = new Tensor(new Float64Array(0), 0, 0);
    expect(t.shape).toEqual([0, 0]);
    expect(t.toArray()).toEqual([]);
  });
  it("desde un arreglo vacío da 0 columnas", () => {
    const t = tensor([]);
    expect(t.shape).toEqual([0, 0]);
    expect(t.toArray()).toEqual([]);
  });

  it("reshape comparte buffer y valida el total", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const r = t.reshape(3, 2);
    expect(r.shape).toEqual([3, 2]);
    expect(r.toArray()).toEqual([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    expect(() => t.reshape(4, 2)).toThrow();
  });

  it("reshape infiere una dimensión con -1", () => {
    const t = tensor([[1, 2, 3, 4]]);
    expect(t.reshape(2, -1).shape).toEqual([2, 2]);
    expect(t.reshape(-1, 1).shape).toEqual([4, 1]);
  });

  it("transpose intercambia filas y columnas", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(t.transpose().toArray()).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
  });

  it("slice y row son vistas", () => {
    const t = tensor([
      [1, 1],
      [2, 2],
      [3, 3],
    ]);
    expect(t.slice(1, 3).toArray()).toEqual([
      [2, 2],
      [3, 3],
    ]);
    expect(t.row(0).toArray()).toEqual([[1, 1]]);
    expect(t.at(2, 1)).toBe(3);
  });
});
