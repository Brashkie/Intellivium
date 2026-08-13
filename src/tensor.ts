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

  // ---- Factories ----

  /** Tensor (rows, cols) lleno de `value`. */
  static full(rows: number, cols: number, value: number): Tensor {
    const data = new Float64Array(rows * cols);
    data.fill(value);
    return new Tensor(data, rows, cols);
  }

  /** Tensor (rows, cols) de ceros. */
  static zeros(rows: number, cols: number): Tensor {
    return new Tensor(new Float64Array(rows * cols), rows, cols);
  }

  /** Tensor (rows, cols) de unos. */
  static ones(rows: number, cols: number): Tensor {
    return Tensor.full(rows, cols, 1);
  }

  /** Identidad (n, n). */
  static eye(n: number): Tensor {
    const t = Tensor.zeros(n, n);
    for (let i = 0; i < n; i++) t.data[i * n + i] = 1;
    return t;
  }

  // ---- Operaciones elemento a elemento ----

  /** Aplica `fn` a cada elemento y devuelve un nuevo tensor. */
  map(fn: (value: number, i: number, j: number) => number): Tensor {
    const out = new Float64Array(this.rows * this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        const idx = i * this.cols + j;
        out[idx] = fn(this.data[idx], i, j);
      }
    }
    return new Tensor(out, this.rows, this.cols);
  }

  /** Copia profunda. */
  clone(): Tensor {
    return new Tensor(Float64Array.from(this.data), this.rows, this.cols);
  }

  /**
   * Combina dos tensores elemento a elemento. Soporta broadcasting cuando el
   * otro tensor es (1, cols) —por filas— o (rows, 1) —por columnas—.
   */
  private zip(other: Tensor, op: (a: number, b: number) => number): Tensor {
    const sameShape = other.rows === this.rows && other.cols === this.cols;
    const rowVec = other.rows === 1 && other.cols === this.cols;
    const colVec = other.cols === 1 && other.rows === this.rows;
    if (!sameShape && !rowVec && !colVec) {
      throw new Error(
        `shapes incompatibles: (${this.rows}, ${this.cols}) vs (${other.rows}, ${other.cols})`,
      );
    }
    return this.map((v, i, j) => {
      const b = rowVec ? other.data[j] : colVec ? other.data[i] : other.data[i * this.cols + j];
      return op(v, b);
    });
  }

  /** Suma elemento a elemento (con broadcasting de vector fila/columna). */
  add(other: Tensor): Tensor {
    return this.zip(other, (a, b) => a + b);
  }

  /** Resta elemento a elemento. */
  sub(other: Tensor): Tensor {
    return this.zip(other, (a, b) => a - b);
  }

  /** Producto de Hadamard (elemento a elemento). */
  mul(other: Tensor): Tensor {
    return this.zip(other, (a, b) => a * b);
  }

  /** Multiplica por un escalar. */
  scale(k: number): Tensor {
    return this.map((v) => v * k);
  }

  /** Suma un escalar a cada elemento. */
  addScalar(k: number): Tensor {
    return this.map((v) => v + k);
  }

  /** Negación elemento a elemento. */
  neg(): Tensor {
    return this.map((v) => -v);
  }

  /** Producto matricial (rows, k) · (k, cols) -> (rows, cols). Utilidad en JS. */
  matmul(other: Tensor): Tensor {
    if (this.cols !== other.rows) {
      throw new Error(
        `matmul: (${this.rows}, ${this.cols}) · (${other.rows}, ${other.cols}) no encajan`,
      );
    }
    const out = new Float64Array(this.rows * other.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let k = 0; k < this.cols; k++) {
        const a = this.data[i * this.cols + k];
        if (a === 0) continue;
        for (let j = 0; j < other.cols; j++) {
          out[i * other.cols + j] += a * other.data[k * other.cols + j];
        }
      }
    }
    return new Tensor(out, this.rows, other.cols);
  }

  // ---- Reducciones ----

  /** Suma de todos los elementos. */
  sum(): number {
    let s = 0;
    for (let i = 0; i < this.data.length; i++) s += this.data[i];
    return s;
  }

  /** Media de todos los elementos. */
  mean(): number {
    return this.data.length === 0 ? 0 : this.sum() / this.data.length;
  }

  /** Máximo. */
  max(): number {
    let m = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < this.data.length; i++) if (this.data[i] > m) m = this.data[i];
    return m;
  }

  /** Mínimo. */
  min(): number {
    let m = Number.POSITIVE_INFINITY;
    for (let i = 0; i < this.data.length; i++) if (this.data[i] < m) m = this.data[i];
    return m;
  }

  /** Índice del máximo por fila (útil en clasificación). */
  argmaxRows(): number[] {
    const out: number[] = [];
    for (let i = 0; i < this.rows; i++) {
      let best = 0;
      for (let j = 1; j < this.cols; j++) {
        if (this.data[i * this.cols + j] > this.data[i * this.cols + best]) best = j;
      }
      out.push(best);
    }
    return out;
  }

  /** Igualdad aproximada (por defecto tol=1e-6). */
  equals(other: Tensor, tol = 1e-6): boolean {
    if (other.rows !== this.rows || other.cols !== this.cols) return false;
    for (let i = 0; i < this.data.length; i++) {
      if (Math.abs(this.data[i] - other.data[i]) > tol) return false;
    }
    return true;
  }

  /** Limita cada elemento al rango [lo, hi]. */
  clip(lo: number, hi: number): Tensor {
    return this.map((v) => (v < lo ? lo : v > hi ? hi : v));
  }

  /** Concatena por filas (axis 0). Requiere el mismo número de columnas. */
  static concatRows(tensors: Tensor[]): Tensor {
    if (tensors.length === 0) throw new Error("concatRows: lista vacía");
    const cols = tensors[0].cols;
    let rows = 0;
    for (const t of tensors) {
      if (t.cols !== cols) throw new Error("concatRows: columnas distintas");
      rows += t.rows;
    }
    const out = new Float64Array(rows * cols);
    let off = 0;
    for (const t of tensors) {
      out.set(t.data, off);
      off += t.data.length;
    }
    return new Tensor(out, rows, cols);
  }

  /** Concatena por columnas (axis 1). Requiere el mismo número de filas. */
  static concatCols(tensors: Tensor[]): Tensor {
    if (tensors.length === 0) throw new Error("concatCols: lista vacía");
    const rows = tensors[0].rows;
    let cols = 0;
    for (const t of tensors) {
      if (t.rows !== rows) throw new Error("concatCols: filas distintas");
      cols += t.cols;
    }
    const out = new Float64Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      let c0 = 0;
      for (const t of tensors) {
        for (let c = 0; c < t.cols; c++) out[r * cols + c0 + c] = t.data[r * t.cols + c];
        c0 += t.cols;
      }
    }
    return new Tensor(out, rows, cols);
  }

  /** Apila filas de igual longitud en un tensor (n, len). */
  static stack(rows: number[][]): Tensor {
    return Tensor.from(rows);
  }

  /** Suma por eje: axis 0 -> (1, cols); axis 1 -> (rows, 1). */
  sumAxis(axis: 0 | 1): Tensor {
    if (axis === 0) {
      const out = new Float64Array(this.cols);
      for (let i = 0; i < this.rows; i++)
        for (let j = 0; j < this.cols; j++) out[j] += this.data[i * this.cols + j];
      return new Tensor(out, 1, this.cols);
    }
    const out = new Float64Array(this.rows);
    for (let i = 0; i < this.rows; i++)
      for (let j = 0; j < this.cols; j++) out[i] += this.data[i * this.cols + j];
    return new Tensor(out, this.rows, 1);
  }

  /** Media por eje (ver {@link sumAxis}). */
  meanAxis(axis: 0 | 1): Tensor {
    const denom = axis === 0 ? this.rows : this.cols;
    return this.sumAxis(axis).scale(denom === 0 ? 0 : 1 / denom);
  }

  /** Softmax por filas (numéricamente estable). Útil para leer probabilidades. */
  softmaxRows(): Tensor {
    const out = new Float64Array(this.rows * this.cols);
    for (let i = 0; i < this.rows; i++) {
      let m = Number.NEGATIVE_INFINITY;
      for (let j = 0; j < this.cols; j++) m = Math.max(m, this.data[i * this.cols + j]);
      let sum = 0;
      for (let j = 0; j < this.cols; j++) {
        const e = Math.exp(this.data[i * this.cols + j] - m);
        out[i * this.cols + j] = e;
        sum += e;
      }
      for (let j = 0; j < this.cols; j++) out[i * this.cols + j] /= sum;
    }
    return new Tensor(out, this.rows, this.cols);
  }
}

/** Helper: crea un tensor a partir de un arreglo 2D. */
export function tensor(arr: number[][]): Tensor {
  return Tensor.from(arr);
}
