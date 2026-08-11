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

  it("factories: zeros, ones, full, eye", () => {
    expect(Tensor.zeros(2, 2).toArray()).toEqual([
      [0, 0],
      [0, 0],
    ]);
    expect(Tensor.ones(1, 3).toArray()).toEqual([[1, 1, 1]]);
    expect(Tensor.full(2, 1, 5).toArray()).toEqual([[5], [5]]);
    expect(Tensor.eye(3).toArray()).toEqual([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]);
  });

  it("add/sub/mul elemento a elemento", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const b = tensor([
      [10, 20],
      [30, 40],
    ]);
    expect(a.add(b).toArray()).toEqual([
      [11, 22],
      [33, 44],
    ]);
    expect(b.sub(a).toArray()).toEqual([
      [9, 18],
      [27, 36],
    ]);
    expect(a.mul(b).toArray()).toEqual([
      [10, 40],
      [90, 160],
    ]);
  });

  it("broadcasting de vector fila y columna", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(a.add(tensor([[10, 20, 30]])).toArray()).toEqual([
      [11, 22, 33],
      [14, 25, 36],
    ]);
    expect(a.add(tensor([[100], [200]])).toArray()).toEqual([
      [101, 102, 103],
      [204, 205, 206],
    ]);
    expect(() => a.add(tensor([[1, 2]]))).toThrow();
  });

  it("scale, addScalar, neg y map", () => {
    const a = tensor([[1, -2, 3]]);
    expect(a.scale(2).toArray()).toEqual([[2, -4, 6]]);
    expect(a.addScalar(1).toArray()).toEqual([[2, -1, 4]]);
    expect(a.neg().toArray()).toEqual([[-1, 2, -3]]);
    expect(a.map((v) => v * v).toArray()).toEqual([[1, 4, 9]]);
  });

  it("matmul con verificación de shapes", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const b = tensor([
      [5, 6],
      [7, 8],
    ]);
    expect(a.matmul(b).toArray()).toEqual([
      [19, 22],
      [43, 50],
    ]);
    expect(() => a.matmul(tensor([[1, 2, 3]]))).toThrow();
    // camino con ceros (optimización a===0)
    const z = tensor([
      [0, 1],
      [2, 0],
    ]);
    expect(
      z
        .matmul(
          tensor([
            [1, 0],
            [0, 1],
          ]),
        )
        .toArray(),
    ).toEqual([
      [0, 1],
      [2, 0],
    ]);
  });

  it("reducciones: sum, mean, max, min, argmaxRows", () => {
    const a = tensor([
      [1, 5, 2],
      [8, 3, 4],
    ]);
    expect(a.sum()).toBe(23);
    expect(a.mean()).toBeCloseTo(23 / 6);
    expect(tensor([]).mean()).toBe(0);
    expect(a.max()).toBe(8);
    expect(a.min()).toBe(1);
    expect(a.argmaxRows()).toEqual([1, 0]);
  });

  it("clone es independiente y equals compara con tolerancia", () => {
    const a = tensor([[1, 2]]);
    const c = a.clone();
    c.data[0] = 99;
    expect(a.at(0, 0)).toBe(1);
    expect(a.equals(tensor([[1, 2]]))).toBe(true);
    expect(a.equals(tensor([[1, 2.0000001]]))).toBe(true);
    expect(a.equals(tensor([[1, 2.1]]))).toBe(false);
    expect(a.equals(tensor([[1, 2, 3]]))).toBe(false);
  });
});
