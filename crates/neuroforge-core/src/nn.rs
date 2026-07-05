//! Capas, modelo secuencial, entrenamiento y predicción.

use ndarray::Array2;

use crate::rng::Rng;
use crate::tape::Tape;

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Linear,
    Relu,
    Sigmoid,
    Tanh,
}

impl Activation {
    pub fn from_str(s: &str) -> Activation {
        match s.to_lowercase().as_str() {
            "relu" => Activation::Relu,
            "sigmoid" => Activation::Sigmoid,
            "tanh" => Activation::Tanh,
            _ => Activation::Linear,
        }
    }
}

/// Función de pérdida.
#[derive(Clone, Copy, Debug)]
pub enum Loss {
    Mse,
    Bce,
}

impl Loss {
    pub fn from_str(s: &str) -> Loss {
        match s.to_lowercase().as_str() {
            "bce" | "binary_crossentropy" | "crossentropy" => Loss::Bce,
            _ => Loss::Mse,
        }
    }
}

/// Optimizador. `Adam` guarda sus hiperparámetros; el estado (momentos) vive
/// en cada capa.
#[derive(Clone, Copy, Debug)]
pub enum Optimizer {
    Sgd,
    Adam { beta1: f32, beta2: f32, eps: f32 },
}

impl Optimizer {
    pub fn adam_default() -> Optimizer {
        Optimizer::Adam { beta1: 0.9, beta2: 0.999, eps: 1e-8 }
    }

    pub fn from_str(s: &str) -> Optimizer {
        match s.to_lowercase().as_str() {
            "adam" => Optimizer::adam_default(),
            _ => Optimizer::Sgd,
        }
    }
}

/// Configuración de entrenamiento.
#[derive(Clone, Copy, Debug)]
pub struct TrainConfig {
    pub epochs: usize,
    pub lr: f32,
    pub loss: Loss,
    pub optimizer: Optimizer,
}

impl TrainConfig {
    pub fn sgd(epochs: usize, lr: f32) -> Self {
        TrainConfig { epochs, lr, loss: Loss::Mse, optimizer: Optimizer::Sgd }
    }

    pub fn adam(epochs: usize, lr: f32) -> Self {
        TrainConfig { epochs, lr, loss: Loss::Mse, optimizer: Optimizer::adam_default() }
    }
}

/// Un paso de Adam sobre un parámetro (actualiza in-place p, m y v).
#[allow(clippy::too_many_arguments)]
fn adam_step(
    p: &mut Array2<f32>,
    m: &mut Array2<f32>,
    v: &mut Array2<f32>,
    g: &Array2<f32>,
    lr: f32,
    b1: f32,
    b2: f32,
    eps: f32,
    t: i32,
) {
    *m = &(&*m * b1) + &(g * (1.0 - b1));
    let g2 = g * g;
    *v = &(&*v * b2) + &(g2 * (1.0 - b2));
    let mhat = &*m / (1.0 - b1.powi(t));
    let vhat = &*v / (1.0 - b2.powi(t));
    let update = mhat / (vhat.mapv(f32::sqrt) + eps);
    *p = &*p - &(update * lr);
}

/// Capa densa (fully-connected): y = act(x . W + b)
pub struct Dense {
    pub w: Array2<f32>, // (in, out)
    pub b: Array2<f32>, // (1, out)
    pub act: Activation,
    // Estado de Adam (sin uso con SGD).
    mw: Array2<f32>,
    vw: Array2<f32>,
    mb: Array2<f32>,
    vb: Array2<f32>,
}

impl Dense {
    pub fn new(inp: usize, out: usize, act: Activation, rng: &mut Rng) -> Self {
        // Inicialización He (buena para relu, decente en general).
        let scale = (2.0 / inp as f32).sqrt();
        let w = Array2::from_shape_fn((inp, out), |_| rng.normal() * scale);
        Dense {
            w,
            b: Array2::zeros((1, out)),
            act,
            mw: Array2::zeros((inp, out)),
            vw: Array2::zeros((inp, out)),
            mb: Array2::zeros((1, out)),
            vb: Array2::zeros((1, out)),
        }
    }

    fn apply_grads(&mut self, gw: &Array2<f32>, gb: &Array2<f32>, opt: &Optimizer, lr: f32, t: i32) {
        match *opt {
            Optimizer::Sgd => {
                self.w = &self.w - &(gw * lr);
                self.b = &self.b - &(gb * lr);
            }
            Optimizer::Adam { beta1, beta2, eps } => {
                adam_step(&mut self.w, &mut self.mw, &mut self.vw, gw, lr, beta1, beta2, eps, t);
                adam_step(&mut self.b, &mut self.mb, &mut self.vb, gb, lr, beta1, beta2, eps, t);
            }
        }
    }
}

pub struct Model {
    pub layers: Vec<Dense>,
    t: i32, // timestep de Adam
}

impl Model {
    pub fn new(layers: Vec<Dense>) -> Self {
        Model { layers, t: 0 }
    }

    /// Construye el grafo forward sobre la cinta y devuelve:
    /// (id_salida, [(id_W, id_b) por capa]) para leer gradientes luego.
    fn forward_tape(&self, tape: &mut Tape, x: usize) -> (usize, Vec<(usize, usize)>) {
        let mut cur = x;
        let mut params = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let wid = tape.leaf(layer.w.clone());
            let bid = tape.leaf(layer.b.clone());
            let z = tape.matmul(cur, wid);
            let z = tape.add(z, bid);
            cur = match layer.act {
                Activation::Linear => z,
                Activation::Relu => tape.relu(z),
                Activation::Sigmoid => tape.sigmoid(z),
                Activation::Tanh => tape.tanh(z),
            };
            params.push((wid, bid));
        }
        (cur, params)
    }

    pub fn predict(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut tape = Tape::new();
        let xid = tape.leaf(x.clone());
        let (out, _) = self.forward_tape(&mut tape, xid);
        tape.value(out).clone()
    }

    /// Entrena según `cfg`. Devuelve el historial de loss por época.
    pub fn train(&mut self, x: &Array2<f32>, y: &Array2<f32>, cfg: &TrainConfig) -> Vec<f32> {
        let mut history = Vec::with_capacity(cfg.epochs);
        for _ in 0..cfg.epochs {
            let mut tape = Tape::new();
            let xid = tape.leaf(x.clone());
            let yid = tape.leaf(y.clone());
            let (out, params) = self.forward_tape(&mut tape, xid);
            let loss = match cfg.loss {
                Loss::Mse => tape.mse(out, yid),
                Loss::Bce => tape.bce(out, yid),
            };
            let loss_val = tape.value(loss)[[0, 0]];

            let grads = tape.backward(loss);
            self.t += 1;
            for (li, (wid, bid)) in params.iter().enumerate() {
                let gw = grads[*wid].clone();
                let gb = grads[*bid].clone();
                self.layers[li].apply_grads(&gw, &gb, &cfg.optimizer, cfg.lr, self.t);
            }
            history.push(loss_val);
        }
        history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn xor_data() -> (Array2<f32>, Array2<f32>) {
        (
            array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            array![[0.0], [1.0], [1.0], [0.0]],
        )
    }

    fn xor_model(rng: &mut Rng) -> Model {
        Model::new(vec![
            Dense::new(2, 8, Activation::Tanh, rng),
            Dense::new(8, 1, Activation::Sigmoid, rng),
        ])
    }

    fn assert_xor(model: &Model, x: &Array2<f32>) {
        let pred = model.predict(x);
        assert!(pred[[0, 0]] < 0.5);
        assert!(pred[[1, 0]] > 0.5);
        assert!(pred[[2, 0]] > 0.5);
        assert!(pred[[3, 0]] < 0.5);
    }

    #[test]
    fn learns_xor_sgd_mse() {
        let mut rng = Rng::new(42);
        let (x, y) = xor_data();
        let mut model = xor_model(&mut rng);
        let hist = model.train(&x, &y, &TrainConfig::sgd(4000, 0.5));
        assert!(*hist.last().unwrap() < 0.05, "loss final demasiado alta");
        assert_xor(&model, &x);
    }

    #[test]
    fn learns_xor_adam_bce() {
        let mut rng = Rng::new(7);
        let (x, y) = xor_data();
        let mut model = xor_model(&mut rng);
        let cfg = TrainConfig {
            epochs: 1500,
            lr: 0.05,
            loss: Loss::Bce,
            optimizer: Optimizer::adam_default(),
        };
        let hist = model.train(&x, &y, &cfg);
        assert!(*hist.last().unwrap() < 0.1, "BCE final demasiado alta: {}", hist.last().unwrap());
        assert_xor(&model, &x);
    }

    #[test]
    fn parsers_work() {
        assert!(matches!(Activation::from_str("relu"), Activation::Relu));
        assert!(matches!(Activation::from_str("otro"), Activation::Linear));
        assert!(matches!(Loss::from_str("bce"), Loss::Bce));
        assert!(matches!(Loss::from_str("mse"), Loss::Mse));
        assert!(matches!(Optimizer::from_str("adam"), Optimizer::Adam { .. }));
        assert!(matches!(Optimizer::from_str("sgd"), Optimizer::Sgd));
    }
}
