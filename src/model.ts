import { type ActivationName, type LayerSpec, dense } from "./layers.js";
import { type NativeModelInstance, getNativeModel } from "./native.js";
import { Tensor } from "./tensor.js";

export type OptimizerName = "sgd" | "adam";
export type LossName = "mse" | "bce" | "cce";

export interface TrainOptions {
  epochs?: number;
  lr?: number;
  /** "sgd" | "adam" (default: "sgd") */
  optimizer?: OptimizerName;
  /** "mse" | "bce" | "cce" (default: "mse") */
  loss?: LossName;
  /** Tamaño de mini-batch. 0/ausente = batch completo. */
  batchSize?: number;
  /** Clipping de gradiente por norma L2 global. 0/ausente = desactivado. */
  gradClip?: number;
  /** Decaimiento del lr por época (lr * lrDecay^epoch). Ausente = 1.0 (sin decaimiento). */
  lrDecay?: number;
  /** Épocas sin mejora antes de parar (early stopping). 0/ausente = desactivado. */
  patience?: number;
  /** Mejora mínima para contar como progreso. */
  minDelta?: number;
  /** Restaurar los pesos de la mejor época al terminar (checkpoint). */
  restoreBest?: boolean;
  /** Hiperparámetros de Adam (opcionales). */
  beta1?: number;
  beta2?: number;
  eps?: number;
}

/** Datos de validación para {@link Model.fit}. */
export interface ValidationData {
  x: Tensor;
  y: Tensor;
}

/** Resultado detallado de {@link Model.fit}. */
export interface TrainOutcome {
  /** Loss de entrenamiento por época. */
  history: number[];
  /** Loss de validación por época (vacío si no se pasó validación). */
  valHistory: number[];
  /** Época con la mejor loss monitorizada. */
  bestEpoch: number;
  /** Mejor loss observada. */
  bestLoss: number;
  /** Si se detuvo por early stopping. */
  stoppedEarly: boolean;
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

  private buildConfig(opts: TrainOptions) {
    return {
      epochs: opts.epochs ?? 100,
      lr: opts.lr ?? 0.01,
      optimizer: opts.optimizer ?? "sgd",
      loss: opts.loss ?? "mse",
      batchSize: opts.batchSize,
      gradClip: opts.gradClip,
      lrDecay: opts.lrDecay,
      patience: opts.patience,
      minDelta: opts.minDelta,
      restoreBest: opts.restoreBest,
      beta1: opts.beta1,
      beta2: opts.beta2,
      eps: opts.eps,
    };
  }

  /** Entrena según las opciones. Resuelve con el historial de loss por época. */
  async train(x: Tensor, y: Tensor, opts: TrainOptions = {}): Promise<number[]> {
    const config = this.buildConfig(opts);
    return this.native.train(x.data, x.rows, x.cols, y.data, y.rows, y.cols, config);
  }

  /**
   * Entrena con validación opcional, early stopping y checkpoint del mejor
   * modelo. Devuelve historiales y metadatos.
   */
  async fit(
    x: Tensor,
    y: Tensor,
    opts: TrainOptions = {},
    validation?: ValidationData,
  ): Promise<TrainOutcome> {
    const config = this.buildConfig(opts);
    return this.native.fit(
      x.data,
      x.rows,
      x.cols,
      y.data,
      y.rows,
      y.cols,
      config,
      validation?.x.data,
      validation?.x.rows,
      validation?.y.data,
      validation?.y.cols,
    );
  }

  /** Calcula la loss sobre un conjunto, sin entrenar. */
  evaluate(x: Tensor, y: Tensor, loss: LossName = "mse"): number {
    return this.native.evaluate(x.data, x.rows, x.cols, y.data, y.cols, loss);
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
