import { describe, expect, it } from "vitest";
import { dense } from "../src/layers.js";
import { tensor } from "../src/tensor.js";

// Estos tests requieren el addon nativo compilado (`npm run build:native`).
// Si no está, se omiten en vez de fallar.
let Model: typeof import("../src/model.js").Model | undefined;
try {
  ({ Model } = await import("../src/model.js"));
} catch {
  Model = undefined;
}

describe.skipIf(!Model)("Model (integración, requiere .node)", () => {
  it("aprende XOR (no lineal)", async () => {
    const X = tensor([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]);
    const y = tensor([[0], [1], [1], [0]]);

    // biome-ignore lint/style/noNonNullAssertion: guard arriba garantiza Model
    const model = new Model!([dense(2, 8, "tanh"), dense(8, 1, "sigmoid")]);

    const history = await model.train(X, y, { epochs: 4000, lr: 0.5 });
    const finalLoss = history.at(-1) ?? Number.POSITIVE_INFINITY;
    expect(finalLoss).toBeLessThan(0.05);

    const pred = model.predict(X).toArray();
    expect(pred[0][0]).toBeLessThan(0.5); // 0 XOR 0 = 0
    expect(pred[1][0]).toBeGreaterThan(0.5); // 0 XOR 1 = 1
    expect(pred[2][0]).toBeGreaterThan(0.5); // 1 XOR 0 = 1
    expect(pred[3][0]).toBeLessThan(0.5); // 1 XOR 1 = 0
  });
});
