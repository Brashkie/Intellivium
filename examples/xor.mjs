// Ejemplo XOR con Adam + BCE, mini-batches y save/load.
// (requiere `npm run build` antes)
import { Model, dense, tensor } from "../lib/index.js";

const X = tensor([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
]);
const y = tensor([[0], [1], [1], [0]]);

const model = new Model([dense(2, 8, "tanh"), dense(8, 1, "sigmoid")]);

const history = await model.train(X, y, {
  epochs: 2000,
  lr: 0.05,
  optimizer: "adam",
  loss: "bce",
  batchSize: 2, // mini-batches
});

console.log("loss final:", history.at(-1).toFixed(5));
console.log(
  "predicciones:",
  model
    .predict(X)
    .toArray()
    .map(([p]) => p.toFixed(3)),
);

// save / load
const state = model.save();
const json = JSON.stringify(state);
const restored = Model.load(JSON.parse(json));
console.log(
  "tras load:  ",
  restored
    .predict(X)
    .toArray()
    .map(([p]) => p.toFixed(3)),
);
