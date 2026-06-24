import type { LayerSpec } from "./layers.js";
import { NativeModel, type NativeModelInstance } from "./native.js";
import { Tensor } from "./tensor.js";

export interface TrainOptions {
  epochs?: number;
  lr?: number;
}

/** Modelo secuencial de capas densas. El cómputo ocurre en el núcleo Rust. */
export class Model {
  private readonly native: NativeModelInstance;

  constructor(layers: LayerSpec[], seed = 42) {
    if (layers.length === 0) {
      throw new Error("el modelo necesita al menos 1 capa");
    }
    this.native = new NativeModel(layers, seed);
  }

  /** Entrena (SGD + MSE). Resuelve con el historial de loss por época. */
  async train(x: Tensor, y: Tensor, opts: TrainOptions = {}): Promise<number[]> {
    const epochs = opts.epochs ?? 100;
    const lr = opts.lr ?? 0.01;
    return this.native.train(x.data, x.rows, x.cols, y.data, y.rows, y.cols, epochs, lr);
  }

  /** Predice salidas para un batch de entradas. */
  predict(x: Tensor): Tensor {
    const out = this.native.predict(x.data, x.rows, x.cols);
    return new Tensor(out, x.rows, this.native.outputDim);
  }
}
