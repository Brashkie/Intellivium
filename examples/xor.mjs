// XOR con Adam + BCE, mini-batches, validación y early stopping.
// (requiere `npm run build` antes)
import { DataLoader, Model, TensorDataset, dense, tensor } from "../lib/index.js";

const X = tensor([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
]);
const y = tensor([[0], [1], [1], [0]]);

const model = new Model([dense(2, 8, "tanh"), dense(8, 1, "sigmoid")]);

// fit con validación + early stopping + checkpoint del mejor modelo
const out = await model.fit(
  X,
  y,
  {
    epochs: 5000,
    lr: 0.05,
    optimizer: "adam",
    loss: "bce",
    batchSize: 2,
    patience: 50,
    restoreBest: true,
  },
  { x: X, y },
);

console.log(`épocas corridas: ${out.history.length} (early stop: ${out.stoppedEarly})`);
console.log(`mejor época: ${out.bestEpoch}  loss: ${out.bestLoss.toFixed(5)}`);
console.log(
  "predicciones:",
  model
    .predict(X)
    .toArray()
    .map(([p]) => p.toFixed(3)),
);

// evaluate sin entrenar
console.log("loss evaluada:", model.evaluate(X, y, "bce").toFixed(5));

// Dataset / DataLoader
const ds = new TensorDataset(X, y);
const loader = new DataLoader(ds, { batchSize: 2, shuffle: true });
console.log(`lotes por época: ${loader.length}`);

// save / load
const restored = Model.load(JSON.parse(JSON.stringify(model.save())));
console.log(
  "tras load:  ",
  restored
    .predict(X)
    .toArray()
    .map(([p]) => p.toFixed(3)),
);
