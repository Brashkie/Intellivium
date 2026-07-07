import { type ActivationName, type LayerSpec, dense } from "./layers.js";
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
  /** Tamaño de mini-batch. 0/ausente = batch completo. */
  batchSize?: number;
  /** Hiperparámetros de Adam (opcionales). */
  beta1?: number;
  beta2?: number;
  eps?: number;
}

/** Estado serializable de una capa. */
export interface LayerState {
  inputDim: number;
  outputDim: number;
  activation: ActivationName;
  weights: number[];
  bias: number[];
}

/** Estado serializable del modelo completo (JSON-friendly). */
export interface ModelState {
  version: 1;
  layers: LayerState[];
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
      batchSize: opts.batchSize,
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

  /** Serializa el modelo a un objeto JSON-friendly (pesos incluidos). */
  save(): ModelState {
    return {
      version: 1,
      layers: this.native.save().map((l) => ({
        inputDim: l.inputDim,
        outputDim: l.outputDim,
        activation: l.activation as ActivationName,
        weights: Array.from(l.weights),
        bias: Array.from(l.bias),
      })),
    };
  }

  /** Reconstruye un modelo desde un estado guardado con {@link Model.save}. */
  static load(state: ModelState, seed = 42): Model {
    const layers = state.layers.map((l) => dense(l.inputDim, l.outputDim, l.activation));
    const model = new Model(layers, seed);
    state.layers.forEach((l, i) => {
      model.native.setWeights(i, Float64Array.from(l.weights), Float64Array.from(l.bias));
    });
    return model;
  }
}
