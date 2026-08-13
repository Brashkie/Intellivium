import { describe, expect, it } from "vitest";
import { TensorDataset } from "../src/data.js";
import { batchNorm, dense, dropout, embedding, layerNorm } from "../src/layers.js";
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
  it("entrena con Dropout + LayerNorm (Fase 3)", async () => {
    const X = tensor([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]);
    const y = tensor([[0], [1], [1], [0]]);
    const model = new Model([
      dense(2, 12, "relu"),
      layerNorm(12),
      dropout(0.1),
      dense(12, 1, "sigmoid"),
    ]);
    const hist = await model.train(X, y, {
      epochs: 2500,
      lr: 0.03,
      optimizer: "adam",
      loss: "bce",
    });
    expect((hist.at(-1) ?? 1) < hist[0]).toBe(true);
  });

  it("save/load conserva capas Dropout/LayerNorm", async () => {
    const X = tensor([
      [0, 0],
      [1, 1],
    ]);
    const y = tensor([[0], [1]]);
    const model = new Model([
      dense(2, 6, "relu"),
      layerNorm(6),
      dropout(0.2),
      dense(6, 1, "sigmoid"),
    ]);
    await model.train(X, y, { epochs: 200, lr: 0.05, optimizer: "adam", loss: "bce" });

    const restored = Model.load(JSON.parse(JSON.stringify(model.save())));
    const a = model.predict(X).toArray();
    const b = restored.predict(X).toArray();
    for (let i = 0; i < a.length; i++) {
      expect(Math.abs(a[i][0] - b[i][0])).toBeLessThan(1e-6);
    }
  });

  it("entrena con BatchNorm y save/load conserva running stats", async () => {
    const X = tensor([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]);
    const y = tensor([[0], [1], [1], [0]]);
    const model = new Model([dense(2, 8, "relu"), batchNorm(8), dense(8, 1, "sigmoid")]);
    const hist = await model.train(X, y, {
      epochs: 1500,
      lr: 0.03,
      optimizer: "adam",
      loss: "bce",
    });
    expect((hist.at(-1) ?? 1) < hist[0]).toBe(true);

    // save/load reproduce exactamente las predicciones (usa running stats)
    const restored = Model.load(JSON.parse(JSON.stringify(model.save())));
    const a = model.predict(X).toArray();
    const b = restored.predict(X).toArray();
    for (let i = 0; i < a.length; i++) {
      expect(Math.abs(a[i][0] - b[i][0])).toBeLessThan(1e-6);
    }
  });

  it("exportWeights/importWeights transfiere pesos entre modelos iguales", async () => {
    const X = tensor([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]);
    const y = tensor([[0], [1], [1], [0]]);
    const trained = new Model([dense(2, 8, "tanh"), dense(8, 1, "sigmoid")]);
    await trained.train(X, y, { epochs: 800, lr: 0.05, optimizer: "adam", loss: "bce" });

    // modelo nuevo (misma arquitectura, init distinto) recibe los pesos
    const fresh = new Model([dense(2, 8, "tanh"), dense(8, 1, "sigmoid")], 999);
    fresh.importWeights(JSON.parse(JSON.stringify(trained.exportWeights())));

    const a = trained.predict(X).toArray();
    const b = fresh.predict(X).toArray();
    for (let i = 0; i < a.length; i++) {
      expect(Math.abs(a[i][0] - b[i][0])).toBeLessThan(1e-6);
    }
  });

  it("importWeights lanza si la arquitectura no coincide", async () => {
    const a = new Model([dense(2, 8, "tanh"), dense(8, 1, "sigmoid")]);
    const b = new Model([dense(2, 4, "tanh"), dense(4, 1, "sigmoid")]);
    expect(() => b.importWeights(a.exportWeights())).toThrow();
  });

  it("importWeights lanza si el número de capas no coincide", async () => {
    const a = new Model([dense(2, 8, "tanh"), dense(8, 1, "sigmoid")]);
    const b = new Model([dense(2, 8, "tanh"), dense(8, 4, "relu"), dense(4, 1, "sigmoid")]);
    // b tiene 3 capas, a exporta 2 -> error de conteo
    expect(() => b.importWeights(a.exportWeights())).toThrow(/capas/);
  });

  it("importWeights lanza si el tipo de capa no coincide", async () => {
    const a = new Model([dense(2, 8, "tanh"), layerNorm(8), dense(8, 1, "sigmoid")]);
    const b = new Model([dense(2, 8, "tanh"), dense(8, 8, "relu"), dense(8, 1, "sigmoid")]);
    // misma cantidad (3) pero capa 1 es layernorm vs dense
    expect(() => b.importWeights(a.exportWeights())).toThrow(/layernorm|dense/);
  });

  it("exportWeights/importWeights transfiere una capa BatchNorm", async () => {
    const X = tensor([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]);
    const y = tensor([[0], [1], [1], [0]]);
    const trained = new Model([dense(2, 8, "relu"), batchNorm(8), dense(8, 1, "sigmoid")]);
    await trained.train(X, y, { epochs: 600, lr: 0.03, optimizer: "adam", loss: "bce" });

    const fresh = new Model([dense(2, 8, "relu"), batchNorm(8), dense(8, 1, "sigmoid")], 123);
    fresh.importWeights(JSON.parse(JSON.stringify(trained.exportWeights())));

    const a = trained.predict(X).toArray();
    const b = fresh.predict(X).toArray();
    for (let i = 0; i < a.length; i++) {
      expect(Math.abs(a[i][0] - b[i][0])).toBeLessThan(1e-6);
    }
  });

  it("Embedding aprende por índice y save/load lo conserva", async () => {
    const x = tensor([[0], [1], [2], [3]]);
    const y = tensor([[0], [1], [1], [0]]);
    const model = new Model([embedding(4, 8), dense(8, 1, "sigmoid")]);
    const hist = await model.train(x, y, {
      epochs: 1500,
      lr: 0.05,
      optimizer: "adam",
      loss: "bce",
    });
    expect(hist.at(-1) ?? 1).toBeLessThan(0.1);

    const restored = Model.load(JSON.parse(JSON.stringify(model.save())));
    const a = model.predict(x).toArray();
    const b = restored.predict(x).toArray();
    for (let i = 0; i < a.length; i++) {
      expect(Math.abs(a[i][0] - b[i][0])).toBeLessThan(1e-6);
    }
  });

  it("exportWeights/importWeights transfiere una capa Embedding", async () => {
    const x = tensor([[0], [1], [2], [3]]);
    const y = tensor([[0], [1], [1], [0]]);
    const trained = new Model([embedding(4, 8), dense(8, 1, "sigmoid")]);
    await trained.train(x, y, { epochs: 600, lr: 0.05, optimizer: "adam", loss: "bce" });

    const fresh = new Model([embedding(4, 8), dense(8, 1, "sigmoid")], 321);
    fresh.importWeights(JSON.parse(JSON.stringify(trained.exportWeights())));

    const a = trained.predict(x).toArray();
    const b = fresh.predict(x).toArray();
    for (let i = 0; i < a.length; i++) {
      expect(Math.abs(a[i][0] - b[i][0])).toBeLessThan(1e-6);
    }
  });
});
