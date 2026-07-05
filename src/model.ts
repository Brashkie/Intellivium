import type { LayerSpec } from "./layers.js";
import { type NativeModelInstance, getNativeModel } from "./native.js";
import { Tensor } from "./tensor.js";

export type OptimizerName = "sgd" | "adam";
export type LossName = "mse" | "bce";

export interface TrainOptions {
  epochs?: number;
  lr?: number;
  /** "sgd" | "adam" (default: "sgd") */
  optimizer?: OptimizerName;
  /** "mse" | "bce" (default: "mse") */
  loss?: LossName;
  /** Hiperparámetros de Adam (opcionales). */
  beta1?: number;
  beta2?: number;
  eps?: number;
}

/** Modelo secuencial de capas densas. El cómputo ocurre en el núcleo Rust. */
export class Model {
  private readonly native: NativeModelInstance;

  constructor(layers: LayerSpec[], seed = 42) {
    if (layers.length === 0) {
      throw new Error("el modelo necesita al menos 1 capa");
    }
    const NativeModel = getNativeModel();
    this.native = new NativeModel(layers, seed);
  }

  /** Entrena según las opciones. Resuelve con el historial de loss por época. */
  async train(x: Tensor, y: Tensor, opts: TrainOptions = {}): Promise<number[]> {
    const config = {
      epochs: opts.epochs ?? 100,
      lr: opts.lr ?? 0.01,
      optimizer: opts.optimizer ?? "sgd",
      loss: opts.loss ?? "mse",
      beta1: opts.beta1,
      beta2: opts.beta2,
      eps: opts.eps,
    };
    return this.native.train(x.data, x.rows, x.cols, y.data, y.rows, y.cols, config);
  }

  /** Predice salidas para un batch de entradas. */
  predict(x: Tensor): Tensor {
    const out = this.native.predict(x.data, x.rows, x.cols);
    return new Tensor(out, x.rows, this.native.outputDim);
  }
}
