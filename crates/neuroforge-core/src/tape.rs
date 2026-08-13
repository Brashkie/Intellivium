//! Motor de autograd (reverse-mode AD) sobre una "tape" (Wengert list).
//!
//! Sin Rc<RefCell>: cada operación registra un nodo en la cinta y `backward`
//! recorre la cinta al revés acumulando gradientes. Todo en f32 / Array2.

use ndarray::{Array2, Axis};

use crate::rng::Rng;

#[derive(Clone)]
enum Op {
    Leaf,
    Add(usize, usize), // soporta broadcast de bias (1, n) sobre (batch, n)
    MatMul(usize, usize),
    Relu(usize),
    Sigmoid(usize),
    Tanh(usize),
    Softmax(usize),                      // softmax por filas (row-wise)
    LeakyRelu(usize),                    // x>0 ? x : 0.01x
    Elu(usize),                          // x>0 ? x : e^x - 1
    Gelu(usize),                         // x * sigmoid(1.702x) (aprox.)
    Mse(usize, usize),                   // pred, target -> escalar (1,1)
    Bce(usize, usize),                   // binary cross-entropy (pred en [0,1]) -> escalar (1,1)
    Cce(usize, usize),                   // categorical cross-entropy (pred=softmax, target=one-hot)
    Mae(usize, usize),                   // mean absolute error (L1)
    Huber(usize, usize),                 // Huber loss (delta=1.0)
    Dropout(usize),                      // input (la máscara se guarda en Tape.masks)
    LayerNorm(usize, usize, usize, f32), // input, gamma, beta, eps
    BatchNorm(usize, usize, usize, f32), // input, gamma, beta, eps (stats por columna)
    Embedding(usize, usize, usize),      // indices_node, table_node, dim
}

const EPS: f32 = 1e-7;
const LEAKY_ALPHA: f32 = 0.01;
const GELU_C: f32 = 1.702;
const HUBER_DELTA: f32 = 1.0;

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Una cinta de cómputo. Cada `Var` es un índice (usize) hacia esta cinta.
pub struct Tape {
    values: Vec<Array2<f32>>,
    ops: Vec<Op>,
    masks: std::collections::HashMap<usize, Array2<f32>>,
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

impl Tape {
    pub fn new() -> Self {
        Tape {
            values: Vec::new(),
            ops: Vec::new(),
            masks: std::collections::HashMap::new(),
        }
    }

    fn push(&mut self, value: Array2<f32>, op: Op) -> usize {
        let id = self.values.len();
        self.values.push(value);
        self.ops.push(op);
        id
    }

    /// Registra un tensor "hoja" (entrada o parámetro).
    pub fn leaf(&mut self, value: Array2<f32>) -> usize {
        self.push(value, Op::Leaf)
    }

    pub fn value(&self, id: usize) -> &Array2<f32> {
        &self.values[id]
    }

    pub fn matmul(&mut self, a: usize, b: usize) -> usize {
        let v = self.values[a].dot(&self.values[b]);
        self.push(v, Op::MatMul(a, b))
    }

    /// Suma con broadcast: si `b` es (1, n) y `a` es (batch, n), se expande.
    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let va = &self.values[a];
        let vb = &self.values[b];
        let v = if vb.shape()[0] == 1 && va.shape()[0] != 1 {
            va + &vb.broadcast(va.raw_dim()).expect("broadcast bias")
        } else {
            va + vb
        };
        self.push(v, Op::Add(a, b))
    }

    pub fn relu(&mut self, a: usize) -> usize {
        let v = self.values[a].mapv(|x| if x > 0.0 { x } else { 0.0 });
        self.push(v, Op::Relu(a))
    }

    pub fn sigmoid(&mut self, a: usize) -> usize {
        let v = self.values[a].mapv(|x| 1.0 / (1.0 + (-x).exp()));
        self.push(v, Op::Sigmoid(a))
    }

    pub fn tanh(&mut self, a: usize) -> usize {
        let v = self.values[a].mapv(|x| x.tanh());
        self.push(v, Op::Tanh(a))
    }

    /// Softmax por filas (numéricamente estable: resta el máximo de cada fila).
    pub fn softmax(&mut self, a: usize) -> usize {
        let z = &self.values[a];
        let mut out = Array2::zeros(z.raw_dim());
        for r in 0..z.nrows() {
            let mut m = f32::NEG_INFINITY;
            for c in 0..z.ncols() {
                m = m.max(z[[r, c]]);
            }
            let mut sum = 0.0;
            for c in 0..z.ncols() {
                let e = (z[[r, c]] - m).exp();
                out[[r, c]] = e;
                sum += e;
            }
            for c in 0..z.ncols() {
                out[[r, c]] /= sum;
            }
        }
        self.push(out, Op::Softmax(a))
    }

    pub fn leaky_relu(&mut self, a: usize) -> usize {
        let v = self.values[a].mapv(|x| if x > 0.0 { x } else { LEAKY_ALPHA * x });
        self.push(v, Op::LeakyRelu(a))
    }

    pub fn elu(&mut self, a: usize) -> usize {
        let v = self.values[a].mapv(|x| if x > 0.0 { x } else { x.exp() - 1.0 });
        self.push(v, Op::Elu(a))
    }

    pub fn gelu(&mut self, a: usize) -> usize {
        let v = self.values[a].mapv(|x| x * sigmoid(GELU_C * x));
        self.push(v, Op::Gelu(a))
    }

    /// Mean Squared Error -> nodo escalar (1,1).
    pub fn mse(&mut self, pred: usize, target: usize) -> usize {
        let diff = &self.values[pred] - &self.values[target];
        let n = diff.len() as f32;
        let loss = (&diff * &diff).sum() / n;
        let v = Array2::from_elem((1, 1), loss);
        self.push(v, Op::Mse(pred, target))
    }

    /// Binary Cross-Entropy -> nodo escalar (1,1). `pred` debe estar en [0,1]
    /// (típicamente salida de sigmoid). Se hace clamp por estabilidad numérica.
    pub fn bce(&mut self, pred: usize, target: usize) -> usize {
        let p = &self.values[pred];
        let t = &self.values[target];
        let n = p.len() as f32;
        let mut acc = 0.0;
        for (&pi, &ti) in p.iter().zip(t.iter()) {
            let pc = pi.clamp(EPS, 1.0 - EPS);
            acc += -(ti * pc.ln() + (1.0 - ti) * (1.0 - pc).ln());
        }
        let v = Array2::from_elem((1, 1), acc / n);
        self.push(v, Op::Bce(pred, target))
    }

    /// Categorical Cross-Entropy -> nodo escalar (1,1). `pred` = salida de softmax,
    /// `target` = one-hot. Promedia por filas (muestras). Clamp por estabilidad.
    pub fn cce(&mut self, pred: usize, target: usize) -> usize {
        let p = &self.values[pred];
        let t = &self.values[target];
        let n = p.nrows() as f32;
        let mut acc = 0.0;
        for (&pi, &ti) in p.iter().zip(t.iter()) {
            let pc = pi.clamp(EPS, 1.0);
            acc += -(ti * pc.ln());
        }
        let v = Array2::from_elem((1, 1), acc / n);
        self.push(v, Op::Cce(pred, target))
    }

    /// Mean Absolute Error (L1) -> escalar (1,1).
    pub fn mae(&mut self, pred: usize, target: usize) -> usize {
        let diff = &self.values[pred] - &self.values[target];
        let n = diff.len() as f32;
        let loss = diff.iter().map(|&d| d.abs()).sum::<f32>() / n;
        self.push(Array2::from_elem((1, 1), loss), Op::Mae(pred, target))
    }

    /// Huber loss (delta=1.0) -> escalar (1,1). Cuadrática cerca de 0, lineal lejos.
    pub fn huber(&mut self, pred: usize, target: usize) -> usize {
        let diff = &self.values[pred] - &self.values[target];
        let n = diff.len() as f32;
        let loss = diff
            .iter()
            .map(|&e| {
                let a = e.abs();
                if a <= HUBER_DELTA {
                    0.5 * e * e
                } else {
                    HUBER_DELTA * (a - 0.5 * HUBER_DELTA)
                }
            })
            .sum::<f32>()
            / n;
        self.push(Array2::from_elem((1, 1), loss), Op::Huber(pred, target))
    }

    /// Inverted dropout: escala por 1/(1-p) las unidades que sobreviven.
    /// La máscara se guarda para el backward. Solo debe usarse en training.
    pub fn dropout(&mut self, a: usize, p: f32, rng: &mut Rng) -> usize {
        let keep = 1.0 - p;
        let scale = if keep > 0.0 { 1.0 / keep } else { 0.0 };
        let mask = self.values[a].mapv(|_| if rng.uniform() < keep { scale } else { 0.0 });
        let y = &self.values[a] * &mask;
        let id = self.push(y, Op::Dropout(a));
        self.masks.insert(id, mask);
        id
    }

    /// Layer Normalization por filas: y = gamma * (x-mean)/sqrt(var+eps) + beta.
    pub fn layer_norm(&mut self, a: usize, gamma: usize, beta: usize, eps: f32) -> usize {
        let (rows, cols) = (self.values[a].nrows(), self.values[a].ncols());
        let cf = cols as f32;
        let mut out = Array2::<f32>::zeros((rows, cols));
        for r in 0..rows {
            let mut mean = 0.0;
            for c in 0..cols {
                mean += self.values[a][[r, c]];
            }
            mean /= cf;
            let mut var = 0.0;
            for c in 0..cols {
                let d = self.values[a][[r, c]] - mean;
                var += d * d;
            }
            var /= cf;
            let std = (var + eps).sqrt();
            for c in 0..cols {
                let xhat = (self.values[a][[r, c]] - mean) / std;
                out[[r, c]] = self.values[gamma][[0, c]] * xhat + self.values[beta][[0, c]];
            }
        }
        self.push(out, Op::LayerNorm(a, gamma, beta, eps))
    }

    /// Batch Normalization por columnas. Normaliza con `mean`/`var` dados
    /// (stats del batch en training, o running en eval), ambos shape (1, features).
    /// El backward recomputa las stats del batch desde la entrada (solo válido
    /// en training, que es cuando se llama a backward).
    pub fn batch_norm(
        &mut self,
        a: usize,
        gamma: usize,
        beta: usize,
        mean: &Array2<f32>,
        var: &Array2<f32>,
        eps: f32,
    ) -> usize {
        let (rows, cols) = (self.values[a].nrows(), self.values[a].ncols());
        let mut out = Array2::<f32>::zeros((rows, cols));
        for c in 0..cols {
            let std = (var[[0, c]] + eps).sqrt();
            for r in 0..rows {
                let xhat = (self.values[a][[r, c]] - mean[[0, c]]) / std;
                out[[r, c]] = self.values[gamma][[0, c]] * xhat + self.values[beta][[0, c]];
            }
        }
        self.push(out, Op::BatchNorm(a, gamma, beta, eps))
    }

    /// Embedding lookup. `idx` es un nodo con índices (batch, L) —valores f32 que
    /// se truncan a usize—, `table` es (vocab, dim). Salida: (batch, L*dim).
    pub fn embedding(&mut self, idx: usize, table: usize, dim: usize) -> usize {
        let (rows, l) = (self.values[idx].nrows(), self.values[idx].ncols());
        let vocab = self.values[table].nrows();
        let mut out = Array2::<f32>::zeros((rows, l * dim));
        for r in 0..rows {
            for li in 0..l {
                let e = (self.values[idx][[r, li]] as usize).min(vocab.saturating_sub(1));
                for d in 0..dim {
                    out[[r, li * dim + d]] = self.values[table][[e, d]];
                }
            }
        }
        self.push(out, Op::Embedding(idx, table, dim))
    }

    /// Backprop desde `out` (típicamente la loss escalar). Devuelve el gradiente
    /// de CADA nodo de la cinta, indexado por su id.
    pub fn backward(&self, out: usize) -> Vec<Array2<f32>> {
        let mut grads: Vec<Array2<f32>> = self
            .values
            .iter()
            .map(|v| Array2::zeros(v.raw_dim()))
            .collect();
        grads[out].fill(1.0);

        for i in (0..self.ops.len()).rev() {
            let g = grads[i].clone();
            match self.ops[i] {
                Op::Leaf => {}
                Op::Add(a, b) => {
                    grads[a] = &grads[a] + &g;
                    if self.values[b].shape()[0] == 1 && g.shape()[0] != 1 {
                        let summed = g.sum_axis(Axis(0)).insert_axis(Axis(0));
                        grads[b] = &grads[b] + &summed;
                    } else {
                        grads[b] = &grads[b] + &g;
                    }
                }
                Op::MatMul(a, b) => {
                    let da = g.dot(&self.values[b].t());
                    let db = self.values[a].t().dot(&g);
                    grads[a] = &grads[a] + &da;
                    grads[b] = &grads[b] + &db;
                }
                Op::Relu(a) => {
                    let mask = self.values[a].mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                    grads[a] = &grads[a] + &(&g * &mask);
                }
                Op::Sigmoid(a) => {
                    let s = &self.values[i];
                    let d = s.mapv(|y| y * (1.0 - y));
                    grads[a] = &grads[a] + &(&g * &d);
                }
                Op::Tanh(a) => {
                    let t = &self.values[i];
                    let d = t.mapv(|y| 1.0 - y * y);
                    grads[a] = &grads[a] + &(&g * &d);
                }
                Op::Softmax(a) => {
                    // dz = s ⊙ (g - rowsum(g ⊙ s))
                    let s = &self.values[i];
                    let mut dz = Array2::zeros(s.raw_dim());
                    for r in 0..s.nrows() {
                        let mut dot = 0.0;
                        for c in 0..s.ncols() {
                            dot += g[[r, c]] * s[[r, c]];
                        }
                        for c in 0..s.ncols() {
                            dz[[r, c]] = s[[r, c]] * (g[[r, c]] - dot);
                        }
                    }
                    grads[a] = &grads[a] + &dz;
                }
                Op::LeakyRelu(a) => {
                    let d = self.values[a].mapv(|x| if x > 0.0 { 1.0 } else { LEAKY_ALPHA });
                    grads[a] = &grads[a] + &(&g * &d);
                }
                Op::Elu(a) => {
                    // x>0 -> 1 ; x<=0 -> e^x
                    let d = self.values[a].mapv(|x| if x > 0.0 { 1.0 } else { x.exp() });
                    grads[a] = &grads[a] + &(&g * &d);
                }
                Op::Gelu(a) => {
                    // f = x*s(cx) ; f' = s(cx) + x*c*s(cx)*(1-s(cx))
                    let d = self.values[a].mapv(|x| {
                        let s = sigmoid(GELU_C * x);
                        s + x * GELU_C * s * (1.0 - s)
                    });
                    grads[a] = &grads[a] + &(&g * &d);
                }
                Op::Mse(p, t) => {
                    let gv = grads[i][[0, 0]];
                    let diff = &self.values[p] - &self.values[t];
                    let n = diff.len() as f32;
                    let dp = diff.mapv(|d| d * 2.0 / n * gv);
                    grads[p] = &grads[p] + &dp;
                }
                Op::Bce(p, t) => {
                    let gv = grads[i][[0, 0]];
                    let pv = &self.values[p];
                    let tv = &self.values[t];
                    let n = pv.len() as f32;
                    let dp = ndarray::Zip::from(pv).and(tv).map_collect(|&pi, &ti| {
                        let pc = pi.clamp(EPS, 1.0 - EPS);
                        (pc - ti) / (pc * (1.0 - pc)) / n * gv
                    });
                    grads[p] = &grads[p] + &dp;
                }
                Op::Cce(p, t) => {
                    let gv = grads[i][[0, 0]];
                    let pv = &self.values[p];
                    let tv = &self.values[t];
                    let n = pv.nrows() as f32;
                    let dp = ndarray::Zip::from(pv).and(tv).map_collect(|&pi, &ti| {
                        let pc = pi.clamp(EPS, 1.0);
                        -(ti / pc) / n * gv
                    });
                    grads[p] = &grads[p] + &dp;
                }
                Op::Mae(p, t) => {
                    let gv = grads[i][[0, 0]];
                    let diff = &self.values[p] - &self.values[t];
                    let n = diff.len() as f32;
                    let dp = diff.mapv(|e| {
                        let s = if e > 0.0 {
                            1.0
                        } else if e < 0.0 {
                            -1.0
                        } else {
                            0.0
                        };
                        s / n * gv
                    });
                    grads[p] = &grads[p] + &dp;
                }
                Op::Huber(p, t) => {
                    let gv = grads[i][[0, 0]];
                    let diff = &self.values[p] - &self.values[t];
                    let n = diff.len() as f32;
                    let dp = diff.mapv(|e| {
                        let g = if e.abs() <= HUBER_DELTA {
                            e
                        } else {
                            HUBER_DELTA * e.signum()
                        };
                        g / n * gv
                    });
                    grads[p] = &grads[p] + &dp;
                }
                Op::Dropout(a) => {
                    let mask = self.masks.get(&i).expect("dropout mask");
                    grads[a] = &grads[a] + &(&g * mask);
                }
                Op::LayerNorm(a, gamma, beta, eps) => {
                    let (rows, cols) = (self.values[a].nrows(), self.values[a].ncols());
                    let cf = cols as f32;
                    let mut dx = Array2::<f32>::zeros((rows, cols));
                    let mut dgamma = Array2::<f32>::zeros((1, cols));
                    let mut dbeta = Array2::<f32>::zeros((1, cols));
                    for r in 0..rows {
                        let mut mean = 0.0;
                        for c in 0..cols {
                            mean += self.values[a][[r, c]];
                        }
                        mean /= cf;
                        let mut var = 0.0;
                        for c in 0..cols {
                            let d = self.values[a][[r, c]] - mean;
                            var += d * d;
                        }
                        var /= cf;
                        let std = (var + eps).sqrt();

                        let mut xhat = vec![0.0f32; cols];
                        let mut dxhat = vec![0.0f32; cols];
                        let mut mean_dxhat = 0.0;
                        let mut mean_dxhat_xhat = 0.0;
                        for c in 0..cols {
                            let xh = (self.values[a][[r, c]] - mean) / std;
                            let dh = g[[r, c]] * self.values[gamma][[0, c]];
                            xhat[c] = xh;
                            dxhat[c] = dh;
                            mean_dxhat += dh;
                            mean_dxhat_xhat += dh * xh;
                            dgamma[[0, c]] += g[[r, c]] * xh;
                            dbeta[[0, c]] += g[[r, c]];
                        }
                        mean_dxhat /= cf;
                        mean_dxhat_xhat /= cf;
                        for c in 0..cols {
                            dx[[r, c]] = (dxhat[c] - mean_dxhat - xhat[c] * mean_dxhat_xhat) / std;
                        }
                    }
                    grads[a] = &grads[a] + &dx;
                    grads[gamma] = &grads[gamma] + &dgamma;
                    grads[beta] = &grads[beta] + &dbeta;
                }
                Op::BatchNorm(a, gamma, beta, eps) => {
                    // Reduce sobre las filas (batch), por columna.
                    let (rows, cols) = (self.values[a].nrows(), self.values[a].ncols());
                    let nf = rows as f32;
                    let mut dx = Array2::<f32>::zeros((rows, cols));
                    let mut dgamma = Array2::<f32>::zeros((1, cols));
                    let mut dbeta = Array2::<f32>::zeros((1, cols));
                    for c in 0..cols {
                        // stats del batch para esta columna
                        let mut mean = 0.0;
                        for r in 0..rows {
                            mean += self.values[a][[r, c]];
                        }
                        mean /= nf;
                        let mut var = 0.0;
                        for r in 0..rows {
                            let d = self.values[a][[r, c]] - mean;
                            var += d * d;
                        }
                        var /= nf;
                        let std = (var + eps).sqrt();

                        let gc = self.values[gamma][[0, c]];
                        let mut sum_dxhat = 0.0;
                        let mut sum_dxhat_xhat = 0.0;
                        for r in 0..rows {
                            let xhat = (self.values[a][[r, c]] - mean) / std;
                            let dxhat = g[[r, c]] * gc;
                            sum_dxhat += dxhat;
                            sum_dxhat_xhat += dxhat * xhat;
                            dgamma[[0, c]] += g[[r, c]] * xhat;
                            dbeta[[0, c]] += g[[r, c]];
                        }
                        let m1 = sum_dxhat / nf;
                        let m2 = sum_dxhat_xhat / nf;
                        for r in 0..rows {
                            let xhat = (self.values[a][[r, c]] - mean) / std;
                            let dxhat = g[[r, c]] * gc;
                            dx[[r, c]] = (dxhat - m1 - xhat * m2) / std;
                        }
                    }
                    grads[a] = &grads[a] + &dx;
                    grads[gamma] = &grads[gamma] + &dgamma;
                    grads[beta] = &grads[beta] + &dbeta;
                }
                Op::Embedding(idx, table, dim) => {
                    // Scatter-add del gradiente hacia las filas de la tabla usadas.
                    let (rows, l) = (self.values[idx].nrows(), self.values[idx].ncols());
                    let vocab = self.values[table].nrows();
                    let mut dtable = Array2::<f32>::zeros((vocab, dim));
                    for r in 0..rows {
                        for li in 0..l {
                            let e =
                                (self.values[idx][[r, li]] as usize).min(vocab.saturating_sub(1));
                            for d in 0..dim {
                                dtable[[e, d]] += g[[r, li * dim + d]];
                            }
                        }
                    }
                    grads[table] = &grads[table] + &dtable;
                    // idx no es diferenciable: su gradiente queda en cero.
                }
            }
        }
        grads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Gradient check numérico de matmul + suma + mse contra diferencias finitas.
    #[test]
    fn matmul_gradient_matches_numeric() {
        let a = array![[1.0_f32, 2.0], [3.0, 4.0]];
        let b = array![[0.5_f32], [-1.0]];
        let target = array![[0.0_f32], [1.0]];

        let analytic = {
            let mut t = Tape::new();
            let ai = t.leaf(a.clone());
            let bi = t.leaf(b.clone());
            let z = t.matmul(ai, bi);
            let ti = t.leaf(target.clone());
            let l = t.mse(z, ti);
            let g = t.backward(l);
            g[bi].clone()
        };

        let eps = 1e-3_f32;
        for i in 0..b.len() {
            let mut bp = b.clone();
            let mut bm = b.clone();
            bp[[i, 0]] += eps;
            bm[[i, 0]] -= eps;
            let loss = |bv: &ndarray::Array2<f32>| {
                let mut t = Tape::new();
                let ai = t.leaf(a.clone());
                let bi = t.leaf(bv.clone());
                let z = t.matmul(ai, bi);
                let ti = t.leaf(target.clone());
                let l = t.mse(z, ti);
                t.value(l)[[0, 0]]
            };
            let numeric = (loss(&bp) - loss(&bm)) / (2.0 * eps);
            assert!(
                (analytic[[i, 0]] - numeric).abs() < 1e-2,
                "grad[{i}]: analytic={} numeric={}",
                analytic[[i, 0]],
                numeric
            );
        }
    }
}
