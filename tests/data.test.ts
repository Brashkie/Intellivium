import { describe, expect, it } from "vitest";
import { DataLoader, TensorDataset } from "../src/data.js";
import { tensor } from "../src/tensor.js";

function makeDataset(n: number): TensorDataset {
  const x = tensor(Array.from({ length: n }, (_, i) => [i, i * 2]));
  const y = tensor(Array.from({ length: n }, (_, i) => [i % 2]));
  return new TensorDataset(x, y);
}

describe("TensorDataset", () => {
  it("expone la cantidad de filas", () => {
    expect(makeDataset(10).length).toBe(10);
  });

  it("rechaza x e y con distinto número de filas", () => {
    const x = tensor([[1], [2], [3]]);
    const y = tensor([[1]]);
    expect(() => new TensorDataset(x, y)).toThrow();
  });

  it("select devuelve las filas pedidas en orden", () => {
    const sub = makeDataset(5).select([3, 0]);
    expect(sub.length).toBe(2);
    expect(sub.x.toArray()).toEqual([
      [3, 6],
      [0, 0],
    ]);
    expect(sub.y.toArray()).toEqual([[1], [0]]);
  });

  it("split reparte todas las filas sin perder ninguna", () => {
    const [train, val] = makeDataset(10).split(0.3);
    expect(val.length).toBe(3);
    expect(train.length).toBe(7);
    expect(train.length + val.length).toBe(10);
  });

  it("split es determinista con el mismo seed", () => {
    const [a] = makeDataset(20).split(0.25, true, 7);
    const [b] = makeDataset(20).split(0.25, true, 7);
    expect(a.x.toArray()).toEqual(b.x.toArray());
  });

  it("split rechaza ratios inválidos", () => {
    expect(() => makeDataset(10).split(0)).toThrow();
    expect(() => makeDataset(10).split(1)).toThrow();
  });
});

describe("DataLoader", () => {
  it("calcula la cantidad de lotes", () => {
    const dl = new DataLoader(makeDataset(10), { batchSize: 4 });
    expect(dl.length).toBe(3); // 4 + 4 + 2
  });

  it("itera cubriendo todas las muestras", () => {
    const dl = new DataLoader(makeDataset(10), { batchSize: 4 });
    let seen = 0;
    for (const batch of dl) {
      expect(batch.x.cols).toBe(2);
      expect(batch.y.cols).toBe(1);
      seen += batch.x.rows;
    }
    expect(seen).toBe(10);
  });

  it("con shuffle sigue cubriendo todas las muestras", () => {
    const dl = new DataLoader(makeDataset(8), { batchSize: 3, shuffle: true, seed: 1 });
    const values: number[] = [];
    for (const batch of dl) {
      for (const row of batch.x.toArray()) {
        values.push(row[0]);
      }
    }
    expect(values.sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  });
  it("usa batchSize 32 por defecto", () => {
    const dl = new DataLoader(makeDataset(40));
    expect(dl.length).toBe(2); // 32 + 8
    let seen = 0;
    for (const batch of dl) seen += batch.x.rows;
    expect(seen).toBe(40);
  });

  it("con shuffle sin seed usa el seed por defecto", () => {
    const a = [...new DataLoader(makeDataset(6), { batchSize: 2, shuffle: true })];
    const b = [...new DataLoader(makeDataset(6), { batchSize: 2, shuffle: true })];
    // mismo seed por defecto -> mismo orden
    expect(a[0].x.toArray()).toEqual(b[0].x.toArray());
  });

  it("acepta seed 0 (cae al estado interno mínimo)", () => {
    const [train, val] = makeDataset(8).split(0.25, true, 0);
    expect(train.length + val.length).toBe(8);
  });
});
