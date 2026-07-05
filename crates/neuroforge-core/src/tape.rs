//! Motor de autograd (reverse-mode AD) sobre una "tape" (Wengert list).
//!
//! Sin Rc<RefCell>: cada operación registra un nodo en la cinta y `backward`
//! recorre la cinta al revés acumulando gradientes. Todo en f32 / Array2.

use ndarray::{Array2, Axis};

#[derive(Clone)]
enum Op {
    Leaf,
    Add(usize, usize), // soporta broadcast de bias (1, n) sobre (batch, n)
    MatMul(usize, usize),
    Relu(usize),
    Sigmoid(usize),
    Tanh(usize),
    Mse(usize, usize), // pred, target -> escalar (1,1)
    Bce(usize, usize), // binary cross-entropy (pred en [0,1]) -> escalar (1,1)
}

const EPS: f32 = 1e-7;

/// Una cinta de cómputo. Cada `Var` es un índice (usize) hacia esta cinta.
pub struct Tape {
    values: Vec<Array2<f32>>,
    ops: Vec<Op>,
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
