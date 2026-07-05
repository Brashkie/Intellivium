// Ejemplo XOR usando el paquete ya compilado (npm run build antes).
// Ahora con Adam + BCE.
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
  epochs: 1500,
  lr: 0.05,
  optimizer: "adam",
  loss: "bce",
});

console.log("optimizador: adam | loss: bce");
console.log("loss inicial:", history[0].toFixed(5));
console.log("loss final  :", history.at(-1).toFixed(5));

const pred = model.predict(X);
console.log("\npredicciones (esperado 0,1,1,0):");
pred.toArray().forEach(([p], i) => {
  console.log(`  ${X.toArray()[i].join(",")} -> ${p.toFixed(4)} (${p > 0.5 ? 1 : 0})`);
});
