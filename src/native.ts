// Carga el addon nativo generado por `napi build --platform` (carpeta ./native).
// index.js resuelve automáticamente el .node correcto según plataforma/arch.
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const native = require("../native/index.js");

export interface NativeLayerSpec {
  inputDim: number;
  outputDim: number;
  activation: string;
}

export interface NativeModelInstance {
  train(
    x: Float64Array,
    xRows: number,
    xCols: number,
    y: Float64Array,
    yRows: number,
    yCols: number,
    epochs: number,
    lr: number,
  ): number[];
  predict(x: Float64Array, xRows: number, xCols: number): Float64Array;
  readonly outputDim: number;
}

export interface NativeModelCtor {
  new (layers: NativeLayerSpec[], seed?: number): NativeModelInstance;
}

export const NativeModel: NativeModelCtor = native.Model;
