/** Tensor 2D (batch x features) respaldado por un Float64Array row-major. */
export class Tensor {
  constructor(
    public readonly data: Float64Array,
    public readonly rows: number,
    public readonly cols: number,
  ) {}

  /** Construye un tensor desde un arreglo 2D, validando que sea rectangular. */
  static from(arr: number[][]): Tensor {
    const rows = arr.length;
    const cols = rows > 0 ? arr[0].length : 0;
    const data = new Float64Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      const row = arr[i];
      if (row.length !== cols) {
        throw new Error(`fila ${i}: se esperaban ${cols} columnas, hay ${row.length}`);
      }
      for (let j = 0; j < cols; j++) {
        data[i * cols + j] = row[j];
      }
    }
    return new Tensor(data, rows, cols);
  }

  /** Devuelve la representación como arreglo 2D. */
  toArray(): number[][] {
    const out: number[][] = [];
    for (let i = 0; i < this.rows; i++) {
      const row: number[] = [];
      for (let j = 0; j < this.cols; j++) {
        row.push(this.data[i * this.cols + j]);
      }
      out.push(row);
    }
    return out;
  }

  get shape(): [number, number] {
    return [this.rows, this.cols];
  }

  /** Elemento (i, j). */
  at(i: number, j: number): number {
    return this.data[i * this.cols + j];
  }

  /**
   * Vista con nueva forma que comparte el mismo buffer (sin copiar).
   * `rows * cols` debe igualar el total de elementos. Usa -1 en una dimensión
   * para inferirla.
   */
  reshape(rows: number, cols: number): Tensor {
    const total = this.rows * this.cols;
    let r = rows;
    let c = cols;
    if (r === -1 && c > 0) r = total / c;
    else if (c === -1 && r > 0) c = total / r;
    if (!Number.isInteger(r) || !Number.isInteger(c) || r * c !== total) {
      throw new Error(`reshape (${rows}, ${cols}) incompatible con ${total} elementos`);
    }
    return new Tensor(this.data, r, c); // comparte el Float64Array (vista)
  }

  /** Transpuesta (copia). */
  transpose(): Tensor {
    const out = new Float64Array(this.rows * this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        out[j * this.rows + i] = this.data[i * this.cols + j];
      }
    }
    return new Tensor(out, this.cols, this.rows);
  }

  /** Vista de filas [start, end) que comparte buffer (subarray). */
  slice(start: number, end: number = this.rows): Tensor {
    const s = Math.max(0, start);
    const e = Math.min(this.rows, end);
    const view = this.data.subarray(s * this.cols, e * this.cols);
    return new Tensor(view, Math.max(0, e - s), this.cols);
  }

  /** Fila i como Tensor (1, cols), compartiendo buffer. */
  row(i: number): Tensor {
    return this.slice(i, i + 1);
  }
}

/** Helper: crea un tensor a partir de un arreglo 2D. */
export function tensor(arr: number[][]): Tensor {
  return Tensor.from(arr);
}
