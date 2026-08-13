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

/** Batch Normalization sobre `features` columnas. */
export interface BatchNormSpec {
  kind: "batchnorm";
  features: number;
}

/** Embedding: tabla entrenable (vocab, dim). Índices -> vectores. */
export interface EmbeddingSpec {
  kind: "embedding";
  vocab: number;
  dim: number;
}

export type LayerSpec = DenseSpec | DropoutSpec | LayerNormSpec | BatchNormSpec | EmbeddingSpec;

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

/** Batch Normalization sobre `features` columnas (running stats, modo train/eval). */
export function batchNorm(features: number): BatchNormSpec {
  return { kind: "batchnorm", features };
}

/** Define una capa de Embedding: embedding(vocab, dim). Entrada (batch, L) de
 * índices -> salida (batch, L*dim). */
export function embedding(vocab: number, dim: number): EmbeddingSpec {
  return { kind: "embedding", vocab, dim };
}
