export type ActivationName =
  | "linear"
  | "relu"
  | "sigmoid"
  | "tanh"
  | "softmax"
  | "leakyrelu"
  | "elu"
  | "gelu";

/** Capa densa: y = act(x·W + b). */
export interface DenseSpec {
  kind: "dense";
  inputDim: number;
  outputDim: number;
  activation: ActivationName;
}

/** Dropout con probabilidad p (activo solo en entrenamiento). */
export interface DropoutSpec {
  kind: "dropout";
  p: number;
}

/** Layer Normalization sobre `features` columnas. */
export interface LayerNormSpec {
  kind: "layernorm";
  features: number;
}

export type LayerSpec = DenseSpec | DropoutSpec | LayerNormSpec;

/** Define una capa densa: dense(entradas, salidas, activacion). */
export function dense(
  inputDim: number,
  outputDim: number,
  activation: ActivationName = "linear",
): DenseSpec {
  return { kind: "dense", inputDim, outputDim, activation };
}

/** Define una capa de dropout: dropout(p). */
export function dropout(p: number): DropoutSpec {
  return { kind: "dropout", p };
}

/** Define una capa de Layer Normalization: layerNorm(features). */
export function layerNorm(features: number): LayerNormSpec {
  return { kind: "layernorm", features };
}
