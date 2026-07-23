// NeuroForge — API pública de JavaScript/TypeScript.
//
// El núcleo numérico vive en Rust (crate neuroforge-core, vía N-API).
// Esta capa solo ofrece ergonomía: tensores, capas y modelo.

export { Tensor, tensor } from "./tensor.js";
export { dense, type ActivationName, type LayerSpec } from "./layers.js";
export { TensorDataset, DataLoader, type Batch } from "./data.js";
export {
  Model,
  type TrainOptions,
  type OptimizerName,
  type LossName,
  type ModelState,
  type LayerState,
  type TrainOutcome,
  type ValidationData,
} from "./model.js";

import { DataLoader, TensorDataset } from "./data.js";
import { dense } from "./layers.js";
import { Model } from "./model.js";
import { Tensor, tensor } from "./tensor.js";

export default { tensor, dense, Model, Tensor, TensorDataset, DataLoader };
