import { describe, expect, it } from "vitest";
import { TensorDataset } from "../src/data.js";
import { dense } from "../src/layers.js";
import { Model } from "../src/model.js";
import { getNativeModel } from "../src/native.js";
import { tensor } from "../src/tensor.js";

// ¿Está compilado el addon nativo? Si no, se omiten los tests de integración.
let nativeAvailable = false;
try {
  getNativeModel();
  nativeAvailable = true;
} catch {
  nativeAvailable = false;
}

describe("Model (validación, sin nativo)", () => {
  it("lanza error si no hay capas", () => {
    expect(() => new Model([])).toThrow();
  });
});

describe.skipIf(!nativeAvailable)("Model (integración, requiere .node)", () => {
  it("aprende XOR con Adam + BCE", async () => {
    const X = tensor([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]);
    const y = tensor([[0], [1], [1], [0]]);

    const model = new Model([dense(2, 8, "tanh"), dense(8, 1, "sigmoid")]);
    const history = await model.train(X, y, {
      epochs: 1500,
      lr: 0.05,
      optimizer: "adam",
      loss: "bce",
    });
    expect(history.at(-1) ?? Number.POSITIVE_INFINITY).toBeLessThan(0.1);

    const pred = model.predict(X).toArray();
    expect(pred[0][0]).toBeLessThan(0.5);
    expect(pred[1][0]).toBeGreaterThan(0.5);
    expect(pred[2][0]).toBeGreaterThan(0.5);
    expect(pred[3][0]).toBeLessThan(0.5);
  });

  it("entrena por mini-batches", async () => {
    const X = tensor([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]);
    const y = tensor([[0], [1], [1], [0]]);
    const model = new Model([dense(2, 8, "tanh"), dense(8, 1, "sigmoid")]);
    const history = await model.train(X, y, {
      epochs: 3000,
      lr: 0.05,
      optimizer: "adam",
      loss: "bce",
      batchSize: 2,
    });
    expect(history.at(-1) ?? Number.POSITIVE_INFINITY).toBeLessThan(0.2);
  });

  it("save/load reproduce las predicciones", async () => {
    const X = tensor([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]);
    const y = tensor([[0], [1], [1], [0]]);
    const model = new Model([dense(2, 8, "tanh"), dense(8, 1, "sigmoid")]);
    await model.train(X, y, { epochs: 800, lr: 0.05, optimizer: "adam", loss: "bce" });

    const state = model.save();
    const json = JSON.stringify(state);
    const restored = Model.load(JSON.parse(json));

    const a = model.predict(X).toArray();
    const b = restored.predict(X).toArray();
    for (let i = 0; i < a.length; i++) {
      expect(Math.abs(a[i][0] - b[i][0])).toBeLessThan(1e-6);
    }
  });

  it("clasifica 3 clases con softmax + cce", async () => {
    const X = tensor([
      [2, 0],
      [-2, 0],
      [0, 2],
      [0, -2],
    ]);
    const y = tensor([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [0, 0, 1],
    ]);
    const model = new Model([dense(2, 12, "relu"), dense(12, 3, "softmax")]);
    await model.train(X, y, { epochs: 2000, lr: 0.05, optimizer: "adam", loss: "cce" });

    const pred = model.predict(X).toArray();
    const argmax = (row: number[]) => row.indexOf(Math.max(...row));
    expect(argmax(pred[0])).toBe(0);
    expect(argmax(pred[1])).toBe(1);
    expect(argmax(pred[2])).toBe(2);
    // cada fila softmax suma ~1
    for (const row of pred) {
      expect(Math.abs(row.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-4);
    }
  });
  it("fit con validación registra valHistory y para temprano", async () => {
    const X = tensor([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]);
    const y = tensor([[0], [1], [1], [0]]);
    const model = new Model([dense(2, 8, "tanh"), dense(8, 1, "sigmoid")]);

    const out = await model.fit(
      X,
      y,
      { epochs: 5000, lr: 0.05, optimizer: "adam", loss: "bce", patience: 25, minDelta: 1e-4 },
      { x: X, y },
    );

    expect(out.valHistory.length).toBe(out.history.length);
    expect(out.history.length).toBeLessThan(5000);
    expect(out.stoppedEarly).toBe(true);
    expect(out.bestLoss).toBeLessThan(out.history[0]);
  });

  it("evaluate devuelve una loss finita", async () => {
    const X = tensor([
      [0, 0],
      [1, 1],
    ]);
    const y = tensor([[0], [0]]);
    const model = new Model([dense(2, 4, "tanh"), dense(4, 1, "sigmoid")]);
    const loss = model.evaluate(X, y, "bce");
    expect(Number.isFinite(loss)).toBe(true);
  });

  it("entrena usando un split de TensorDataset", async () => {
    const rows = Array.from({ length: 40 }, (_, i) => [i % 2, (i + 1) % 2]);
    const labels = rows.map((r) => [r[0]]);
    const ds = new TensorDataset(tensor(rows), tensor(labels));
    const [train, val] = ds.split(0.25);

    const model = new Model([dense(2, 6, "relu"), dense(6, 1, "sigmoid")]);
    const out = await model.fit(
      train.x,
      train.y,
      { epochs: 300, lr: 0.05, optimizer: "adam", loss: "bce", restoreBest: true },
      { x: val.x, y: val.y },
    );
    expect(out.valHistory.length).toBe(out.history.length);
    expect(out.bestEpoch).toBeGreaterThanOrEqual(0);
  });
  it("train sin opciones usa los valores por defecto", async () => {
    const X = tensor([
      [0, 0],
      [1, 1],
    ]);
    const y = tensor([[0], [1]]);
    const model = new Model([dense(2, 4, "tanh"), dense(4, 1, "sigmoid")]);
    // sin opts: epochs=100, lr=0.01, optimizer="sgd", loss="mse"
    const history = await model.train(X, y);
    expect(history.length).toBe(100);
    expect(Number.isFinite(history.at(-1) as number)).toBe(true);
  });

  it("fit sin opciones ni validación devuelve valHistory vacío", async () => {
    const X = tensor([
      [0, 0],
      [1, 1],
    ]);
    const y = tensor([[0], [1]]);
    const model = new Model([dense(2, 4, "tanh"), dense(4, 1, "sigmoid")]);
    const out = await model.fit(X, y);
    expect(out.history.length).toBe(100);
    expect(out.valHistory).toEqual([]);
    expect(out.stoppedEarly).toBe(false);
  });

  it("evaluate usa mse por defecto", () => {
    const X = tensor([
      [0, 0],
      [1, 1],
    ]);
    const y = tensor([[0], [1]]);
    const model = new Model([dense(2, 4, "tanh"), dense(4, 1, "sigmoid")]);
    expect(Number.isFinite(model.evaluate(X, y))).toBe(true);
  });
});
