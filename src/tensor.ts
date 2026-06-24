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
}

/** Helper: crea un tensor a partir de un arreglo 2D. */
export function tensor(arr: number[][]): Tensor {
  return Tensor.from(arr);
}
