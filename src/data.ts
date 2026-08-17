import { Tensor } from "./tensor.js";

/** Un par (entrada, objetivo) ya en forma de tensores. */
export interface Batch {
  x: Tensor;
  y: Tensor;
}

/**
 * Interfaz de dataset. Cualquier fuente de datos (en memoria, generada,
 * perezosa, leída de disco…) que exponga `length` y `get(i)` sirve como
 * dataset para {@link DataLoader}.
 */
export interface Dataset {
  /** Número de muestras. */
  readonly length: number;
  /** Devuelve la muestra `index` como (x, y), cada uno un arreglo plano. */
  get(index: number): { x: number[]; y: number[] };
}

/**
 * Dataset en memoria respaldado por dos tensores alineados por filas.
 * Es la base para {@link DataLoader} y para dividir en train/validación.
 */
export class TensorDataset implements Dataset {
  constructor(
    public readonly x: Tensor,
    public readonly y: Tensor,
  ) {
    if (x.rows !== y.rows) {
      throw new Error(`x tiene ${x.rows} filas pero y tiene ${y.rows}`);
    }
  }

  get length(): number {
    return this.x.rows;
  }

  /** Devuelve la muestra `index` como (x, y) planos. */
  get(index: number): { x: number[]; y: number[] } {
    if (index < 0 || index >= this.length) {
      throw new Error(`índice ${index} fuera de rango [0, ${this.length})`);
    }
    const x = Array.from(this.x.data.subarray(index * this.x.cols, (index + 1) * this.x.cols));
    const y = Array.from(this.y.data.subarray(index * this.y.cols, (index + 1) * this.y.cols));
    return { x, y };
  }

  /** Devuelve un subconjunto con las filas indicadas (en ese orden). */
  select(indices: number[]): TensorDataset {
    const xd = new Float64Array(indices.length * this.x.cols);
    const yd = new Float64Array(indices.length * this.y.cols);
    indices.forEach((src, dst) => {
      for (let c = 0; c < this.x.cols; c++) {
        xd[dst * this.x.cols + c] = this.x.data[src * this.x.cols + c];
      }
      for (let c = 0; c < this.y.cols; c++) {
        yd[dst * this.y.cols + c] = this.y.data[src * this.y.cols + c];
      }
    });
    return new TensorDataset(
      new Tensor(xd, indices.length, this.x.cols),
      new Tensor(yd, indices.length, this.y.cols),
    );
  }

  /**
   * Divide en (train, validación). `ratio` es la fracción de validación.
   * Con `shuffle` baraja antes de cortar (determinista si pasas `seed`).
   */
  split(ratio = 0.2, shuffle = true, seed = 42): [TensorDataset, TensorDataset] {
    if (ratio <= 0 || ratio >= 1) {
      throw new Error("ratio debe estar entre 0 y 1 (exclusivo)");
    }
    let idx = Array.from({ length: this.length }, (_, i) => i);
    if (shuffle) {
      idx = shuffleIndices(idx, seed);
    }
    const nVal = Math.max(1, Math.round(this.length * ratio));
    const valIdx = idx.slice(0, nVal);
    const trainIdx = idx.slice(nVal);
    return [this.select(trainIdx), this.select(valIdx)];
  }
}

/** Itera cualquier {@link Dataset} en lotes, opcionalmente barajado. */
export class DataLoader implements Iterable<Batch> {
  constructor(
    private readonly dataset: Dataset,
    private readonly options: { batchSize?: number; shuffle?: boolean; seed?: number } = {},
  ) {
    if (
      options.batchSize !== undefined &&
      (!Number.isInteger(options.batchSize) || options.batchSize < 1)
    ) {
      throw new Error(`batchSize debe ser un entero >= 1, recibí ${options.batchSize}`);
    }
  }

  /** Número de lotes por pasada completa. */
  get length(): number {
    const bs = this.options.batchSize ?? 32;
    return Math.ceil(this.dataset.length / bs);
  }

  *[Symbol.iterator](): Iterator<Batch> {
    const bs = this.options.batchSize ?? 32;
    let idx = Array.from({ length: this.dataset.length }, (_, i) => i);
    if (this.options.shuffle) {
      idx = shuffleIndices(idx, this.options.seed ?? 42);
    }
    for (let start = 0; start < idx.length; start += bs) {
      const slice = idx.slice(start, start + bs);
      const first = this.dataset.get(slice[0]);
      const xCols = first.x.length;
      const yCols = first.y.length;
      const xd = new Float64Array(slice.length * xCols);
      const yd = new Float64Array(slice.length * yCols);
      slice.forEach((src, row) => {
        const sample = this.dataset.get(src);
        xd.set(sample.x, row * xCols);
        yd.set(sample.y, row * yCols);
      });
      yield {
        x: new Tensor(xd, slice.length, xCols),
        y: new Tensor(yd, slice.length, yCols),
      };
    }
  }
}

/** Fisher-Yates determinista con un LCG simple (mismo seed → mismo orden). */
function shuffleIndices(indices: number[], seed: number): number[] {
  const out = indices.slice();
  let state = seed >>> 0 || 1;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x1_0000_0000;
  };
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}
