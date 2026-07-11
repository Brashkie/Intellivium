// Carga perezosa del addon nativo generado por `napi build --platform`.
// El binding se emite como `index.cjs` (CommonJS) en la raíz para no chocar
// con "type": "module". Se resuelve solo al construir el primer Model.
import { createRequire } from "node:module";

export interface NativeLayerSpec {
  inputDim: number;
  outputDim: number;
  activation: string;
}

export interface NativeTrainConfig {
  epochs: number;
  lr: number;
  optimizer?: string;
  loss?: string;
  beta1?: number;
  beta2?: number;
  eps?: number;
  batchSize?: number;
  gradClip?: number;
  lrDecay?: number;
}

export interface NativeLayerState {
  inputDim: number;
  outputDim: number;
  activation: string;
  weights: Float64Array;
  bias: Float64Array;
}

export interface NativeModelInstance {
  train(
    x: Float64Array,
    xRows: number,
    xCols: number,
    y: Float64Array,
    yRows: number,
    yCols: number,
    config: NativeTrainConfig,
  ): number[];
  predict(x: Float64Array, xRows: number, xCols: number): Float64Array;
  save(): NativeLayerState[];
  setWeights(index: number, weights: Float64Array, bias: Float64Array): void;
  readonly outputDim: number;
}

export interface NativeModelCtor {
  new (layers: NativeLayerSpec[], seed?: number): NativeModelInstance;
}

interface NativeModule {
  Model: NativeModelCtor;
}

let cached: NativeModule | undefined;

/** Devuelve el constructor nativo `Model`, cargando el addon una sola vez. */
export function getNativeModel(): NativeModelCtor {
  if (!cached) {
    const require = createRequire(import.meta.url);
    cached = require("../index.cjs") as NativeModule;
  }
  return cached.Model;
}
