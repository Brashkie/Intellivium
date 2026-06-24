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

/// Capa densa (fully-connected): y = act(x . W + b)
pub struct Dense {
    pub w: Array2<f32>, // (in, out)
    pub b: Array2<f32>, // (1, out)
    pub act: Activation,
}

impl Dense {
    pub fn new(inp: usize, out: usize, act: Activation, rng: &mut Rng) -> Self {
        // Inicialización He (buena para relu, decente en general).
        let scale = (2.0 / inp as f32).sqrt();
        let w = Array2::from_shape_fn((inp, out), |_| rng.normal() * scale);
        let b = Array2::zeros((1, out));
        Dense { w, b, act }
    }
}

pub struct Model {
    pub layers: Vec<Dense>,
}

impl Model {
    pub fn new(layers: Vec<Dense>) -> Self {
        Model { layers }
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

    /// Entrena con SGD plano + MSE. Devuelve el historial de loss por época.
    pub fn train(
        &mut self,
        x: &Array2<f32>,
        y: &Array2<f32>,
        epochs: usize,
        lr: f32,
    ) -> Vec<f32> {
        let mut history = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            let mut tape = Tape::new();
            let xid = tape.leaf(x.clone());
            let yid = tape.leaf(y.clone());
            let (out, params) = self.forward_tape(&mut tape, xid);
            let loss = tape.mse(out, yid);
            let loss_val = tape.value(loss)[[0, 0]];

            let grads = tape.backward(loss);
            for (li, (wid, bid)) in params.iter().enumerate() {
                let gw = &grads[*wid];
                let gb = &grads[*bid];
                self.layers[li].w = &self.layers[li].w - &(gw * lr);
                self.layers[li].b = &self.layers[li].b - &(gb * lr);
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

    #[test]
    fn learns_xor() {
        let mut rng = Rng::new(42);
        let x = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let y = array![[0.0], [1.0], [1.0], [0.0]];
        let mut model = Model::new(vec![
            Dense::new(2, 8, Activation::Tanh, &mut rng),
            Dense::new(8, 1, Activation::Sigmoid, &mut rng),
        ]);
        let hist = model.train(&x, &y, 4000, 0.5);
        assert!(*hist.last().unwrap() < 0.05, "loss final demasiado alta");

        let pred = model.predict(&x);
        assert!(pred[[0, 0]] < 0.5);
        assert!(pred[[1, 0]] > 0.5);
        assert!(pred[[2, 0]] > 0.5);
        assert!(pred[[3, 0]] < 0.5);
    }

    #[test]
    fn activation_from_str_parses() {
        assert!(matches!(Activation::from_str("relu"), Activation::Relu));
        assert!(matches!(Activation::from_str("SIGMOID"), Activation::Sigmoid));
        assert!(matches!(Activation::from_str("tanh"), Activation::Tanh));
        assert!(matches!(Activation::from_str("otro"), Activation::Linear));
    }
}
