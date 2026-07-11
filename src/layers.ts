export type ActivationName = "linear" | "relu" | "sigmoid" | "tanh" | "softmax";

export interface LayerSpec {
  inputDim: number;
  outputDim: number;
  activation: ActivationName;
}

/** Define una capa densa: dense(entradas, salidas, activacion). */
export function dense(
  inputDim: number,
  outputDim: number,
  activation: ActivationName = "linear",
): LayerSpec {
  return { inputDim, outputDim, activation };
}
