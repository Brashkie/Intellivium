//! Capas, modelo secuencial, entrenamiento y predicción.

use ndarray::{Array2, Axis};

use crate::rng::Rng;
use crate::tape::Tape;

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Linear,
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
}

impl Activation {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Activation {
        match s.to_lowercase().as_str() {
            "relu" => Activation::Relu,
            "sigmoid" => Activation::Sigmoid,
            "tanh" => Activation::Tanh,
            "softmax" => Activation::Softmax,
            _ => Activation::Linear,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Activation::Linear => "linear",
            Activation::Relu => "relu",
            Activation::Sigmoid => "sigmoid",
            Activation::Tanh => "tanh",
            Activation::Softmax => "softmax",
        }
    }
}

/// Función de pérdida.
#[derive(Clone, Copy, Debug)]
pub enum Loss {
    Mse,
    Bce,
    Cce,
}

impl Loss {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Loss {
        match s.to_lowercase().as_str() {
            "bce" | "binary_crossentropy" => Loss::Bce,
            "cce" | "categorical_crossentropy" | "crossentropy" => Loss::Cce,
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
        Optimizer::Adam {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    #[allow(clippy::should_implement_trait)]
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
    /// Tamaño de mini-batch. 0 = batch completo (todo el dataset por época).
    pub batch_size: usize,
    /// Clipping de gradiente por norma L2 global. 0 = desactivado.
    pub grad_clip: f32,
    /// Decaimiento exponencial del lr por época: lr_e = lr * lr_decay^epoch. 1.0 = sin decaimiento.
    pub lr_decay: f32,
    /// Épocas sin mejora antes de parar (early stopping). 0 = desactivado.
    pub patience: usize,
    /// Mejora mínima para considerar que hubo progreso.
    pub min_delta: f32,
    /// Al terminar, restaurar los pesos de la mejor época (checkpoint).
    pub restore_best: bool,
}

/// Resultado de un entrenamiento con validación.
#[derive(Clone, Debug)]
pub struct TrainResult {
    /// Loss de entrenamiento por época.
    pub history: Vec<f32>,
    /// Loss de validación por época (vacío si no se pasó set de validación).
    pub val_history: Vec<f32>,
    /// Índice de la mejor época (por val loss, o train loss si no hay validación).
    pub best_epoch: usize,
    /// Mejor loss observada.
    pub best_loss: f32,
    /// Si el entrenamiento se detuvo por early stopping.
    pub stopped_early: bool,
}

impl TrainConfig {
    pub fn sgd(epochs: usize, lr: f32) -> Self {
        TrainConfig {
            epochs,
            lr,
            loss: Loss::Mse,
            optimizer: Optimizer::Sgd,
            batch_size: 0,
            grad_clip: 0.0,
            lr_decay: 1.0,
            patience: 0,
            min_delta: 0.0,
            restore_best: false,
        }
    }

    pub fn adam(epochs: usize, lr: f32) -> Self {
        TrainConfig {
            epochs,
            lr,
            loss: Loss::Mse,
            optimizer: Optimizer::adam_default(),
            batch_size: 0,
            grad_clip: 0.0,
            lr_decay: 1.0,
            patience: 0,
            min_delta: 0.0,
            restore_best: false,
        }
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

    fn apply_grads(
        &mut self,
        gw: &Array2<f32>,
        gb: &Array2<f32>,
        opt: &Optimizer,
        lr: f32,
        t: i32,
    ) {
        match *opt {
            Optimizer::Sgd => {
                self.w = &self.w - &(gw * lr);
                self.b = &self.b - &(gb * lr);
            }
            Optimizer::Adam { beta1, beta2, eps } => {
                adam_step(
                    &mut self.w,
                    &mut self.mw,
                    &mut self.vw,
                    gw,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    t,
                );
                adam_step(
                    &mut self.b,
                    &mut self.mb,
                    &mut self.vb,
                    gb,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    t,
                );
            }
        }
    }
}

pub struct Model {
    pub layers: Vec<Dense>,
    t: i32,   // timestep de Adam
    rng: Rng, // para barajar los mini-batches
}

impl Model {
    pub fn new(layers: Vec<Dense>) -> Self {
        Model {
            layers,
            t: 0,
            rng: Rng::new(0x1234_5678),
        }
    }

    /// Reemplaza los pesos de una capa (para load). Resetea el estado de Adam.
    pub fn set_weights(&mut self, i: usize, w: Array2<f32>, b: Array2<f32>) {
        let l = &mut self.layers[i];
        l.mw = Array2::zeros(w.raw_dim());
        l.vw = Array2::zeros(w.raw_dim());
        l.mb = Array2::zeros(b.raw_dim());
        l.vb = Array2::zeros(b.raw_dim());
        l.w = w;
        l.b = b;
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
                Activation::Softmax => tape.softmax(z),
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

    /// Un paso de entrenamiento sobre un batch: forward, backward y update.
    /// Aplica clipping por norma global si `cfg.grad_clip > 0`. Devuelve la loss.
    fn step(&mut self, xb: &Array2<f32>, yb: &Array2<f32>, cfg: &TrainConfig, lr: f32) -> f32 {
        let mut tape = Tape::new();
        let xid = tape.leaf(xb.clone());
        let yid = tape.leaf(yb.clone());
        let (out, params) = self.forward_tape(&mut tape, xid);
        let loss = match cfg.loss {
            Loss::Mse => tape.mse(out, yid),
            Loss::Bce => tape.bce(out, yid),
            Loss::Cce => tape.cce(out, yid),
        };
        let loss_val = tape.value(loss)[[0, 0]];

        let grads = tape.backward(loss);
        // Recoge los gradientes de los parámetros (W, b) por capa.
        let mut pgrads: Vec<(Array2<f32>, Array2<f32>)> = params
            .iter()
            .map(|(wid, bid)| (grads[*wid].clone(), grads[*bid].clone()))
            .collect();

        // Gradient clipping por norma L2 global.
        if cfg.grad_clip > 0.0 {
            let mut sq = 0.0f32;
            for (gw, gb) in &pgrads {
                sq += gw.iter().map(|&v| v * v).sum::<f32>();
                sq += gb.iter().map(|&v| v * v).sum::<f32>();
            }
            let norm = sq.sqrt();
            if norm > cfg.grad_clip {
                let scale = cfg.grad_clip / (norm + 1e-12);
                for (gw, gb) in &mut pgrads {
                    gw.mapv_inplace(|v| v * scale);
                    gb.mapv_inplace(|v| v * scale);
                }
            }
        }

        self.t += 1;
        for (li, (gw, gb)) in pgrads.iter().enumerate() {
            self.layers[li].apply_grads(gw, gb, &cfg.optimizer, lr, self.t);
        }
        loss_val
    }

    /// Calcula la loss sobre un conjunto sin actualizar pesos.
    pub fn evaluate(&self, x: &Array2<f32>, y: &Array2<f32>, loss: Loss) -> f32 {
        let mut tape = Tape::new();
        let xid = tape.leaf(x.clone());
        let yid = tape.leaf(y.clone());
        let (out, _) = self.forward_tape(&mut tape, xid);
        let l = match loss {
            Loss::Mse => tape.mse(out, yid),
            Loss::Bce => tape.bce(out, yid),
            Loss::Cce => tape.cce(out, yid),
        };
        tape.value(l)[[0, 0]]
    }

    /// Copia los pesos actuales (checkpoint en memoria).
    fn snapshot(&self) -> Vec<(Array2<f32>, Array2<f32>)> {
        self.layers
            .iter()
            .map(|l| (l.w.clone(), l.b.clone()))
            .collect()
    }

    /// Restaura pesos desde un snapshot.
    fn restore(&mut self, snap: Vec<(Array2<f32>, Array2<f32>)>) {
        for (i, (w, b)) in snap.into_iter().enumerate() {
            self.layers[i].w = w;
            self.layers[i].b = b;
        }
    }

    /// Entrena con validación opcional, early stopping y checkpoint del mejor
    /// modelo. Si `val` es `None` el criterio de mejora usa la loss de train.
    pub fn train_with_validation(
        &mut self,
        x: &Array2<f32>,
        y: &Array2<f32>,
        val: Option<(&Array2<f32>, &Array2<f32>)>,
        cfg: &TrainConfig,
    ) -> TrainResult {
        let n = x.nrows();
        let bs = if cfg.batch_size == 0 || cfg.batch_size >= n {
            n
        } else {
            cfg.batch_size
        };

        let mut history = Vec::with_capacity(cfg.epochs);
        let mut val_history = Vec::new();
        let mut idx: Vec<usize> = (0..n).collect();

        let mut best_loss = f32::INFINITY;
        let mut best_epoch = 0usize;
        let mut best_snap: Option<Vec<(Array2<f32>, Array2<f32>)>> = None;
        let mut since_improve = 0usize;
        let mut stopped_early = false;

        for epoch in 0..cfg.epochs {
            let lr = cfg.lr * cfg.lr_decay.powi(epoch as i32);

            if bs < n {
                for i in (1..n).rev() {
                    let j = self.rng.usize_below(i + 1);
                    idx.swap(i, j);
                }
            }

            let mut epoch_loss = 0.0f32;
            let mut start = 0;
            while start < n {
                let end = (start + bs).min(n);
                let batch = &idx[start..end];
                let xb = x.select(Axis(0), batch);
                let yb = y.select(Axis(0), batch);
                let lv = self.step(&xb, &yb, cfg, lr);
                epoch_loss += lv * (end - start) as f32;
                start = end;
            }
            let train_loss = epoch_loss / n as f32;
            history.push(train_loss);

            // Criterio de mejora: val loss si hay validación, si no train loss.
            let monitor = match val {
                Some((vx, vy)) => {
                    let vl = self.evaluate(vx, vy, cfg.loss);
                    val_history.push(vl);
                    vl
                }
                None => train_loss,
            };

            if monitor < best_loss - cfg.min_delta {
                best_loss = monitor;
                best_epoch = epoch;
                since_improve = 0;
                if cfg.restore_best {
                    best_snap = Some(self.snapshot());
                }
            } else {
                since_improve += 1;
                if cfg.patience > 0 && since_improve >= cfg.patience {
                    stopped_early = true;
                    break;
                }
            }
        }

        if cfg.restore_best {
            if let Some(snap) = best_snap {
                self.restore(snap);
            }
        }

        TrainResult {
            history,
            val_history,
            best_epoch,
            best_loss,
            stopped_early,
        }
    }

    /// Entrena según `cfg`. Con `batch_size > 0` usa mini-batches barajados por
    /// época; con 0 usa el batch completo. Devuelve la loss media por época.
    pub fn train(&mut self, x: &Array2<f32>, y: &Array2<f32>, cfg: &TrainConfig) -> Vec<f32> {
        self.train_with_validation(x, y, None, cfg).history
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
            batch_size: 0,
            grad_clip: 0.0,
            lr_decay: 1.0,
            patience: 0,
            min_delta: 0.0,
            restore_best: false,
        };
        let hist = model.train(&x, &y, &cfg);
        assert!(
            *hist.last().unwrap() < 0.1,
            "BCE final demasiado alta: {}",
            hist.last().unwrap()
        );
        assert_xor(&model, &x);
    }

    #[test]
    fn parsers_work() {
        assert!(matches!(Activation::from_str("relu"), Activation::Relu));
        assert!(matches!(Activation::from_str("otro"), Activation::Linear));
        assert!(matches!(Loss::from_str("bce"), Loss::Bce));
        assert!(matches!(Loss::from_str("mse"), Loss::Mse));
        assert!(matches!(
            Optimizer::from_str("adam"),
            Optimizer::Adam { .. }
        ));
        assert!(matches!(Optimizer::from_str("sgd"), Optimizer::Sgd));
        assert!(matches!(
            Activation::from_str("softmax"),
            Activation::Softmax
        ));
        assert!(matches!(Loss::from_str("cce"), Loss::Cce));
    }

    #[test]
    fn learns_xor_minibatch() {
        let mut rng = Rng::new(7);
        let (x, y) = xor_data();
        let mut model = xor_model(&mut rng);
        let cfg = TrainConfig {
            epochs: 3000,
            lr: 0.05,
            loss: Loss::Bce,
            optimizer: Optimizer::adam_default(),
            batch_size: 2, // mini-batches de 2 sobre 4 muestras
            grad_clip: 5.0,
            lr_decay: 1.0,
            patience: 0,
            min_delta: 0.0,
            restore_best: false,
        };
        let hist = model.train(&x, &y, &cfg);
        assert!(
            *hist.last().unwrap() < 0.15,
            "minibatch loss final: {}",
            hist.last().unwrap()
        );
        assert_xor(&model, &x);
    }

    #[test]
    fn set_weights_roundtrip() {
        let mut rng = Rng::new(1);
        let (x, y) = xor_data();
        let mut trained = xor_model(&mut rng);
        trained.train(&x, &y, &TrainConfig::adam(1500, 0.05));
        let before = trained.predict(&x);

        // Clonar pesos a un modelo nuevo (misma arquitectura, init distinto).
        let mut rng2 = Rng::new(999);
        let mut restored = xor_model(&mut rng2);
        for i in 0..trained.layers.len() {
            restored.set_weights(i, trained.layers[i].w.clone(), trained.layers[i].b.clone());
        }
        let after = restored.predict(&x);

        for r in 0..before.nrows() {
            assert!(
                (before[[r, 0]] - after[[r, 0]]).abs() < 1e-6,
                "mismatch fila {r}"
            );
        }
    }

    #[test]
    fn learns_3class_softmax_cce() {
        // 4 puntos, 3 clases separables. Salida softmax + loss CCE.
        let mut rng = Rng::new(3);
        let x = array![[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0]];
        // clases: 0, 1, 2, 2 (one-hot)
        let y = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ];
        let mut model = Model::new(vec![
            Dense::new(2, 12, Activation::Relu, &mut rng),
            Dense::new(12, 3, Activation::Softmax, &mut rng),
        ]);
        let cfg = TrainConfig {
            epochs: 2000,
            lr: 0.05,
            loss: Loss::Cce,
            optimizer: Optimizer::adam_default(),
            batch_size: 0,
            grad_clip: 0.0,
            lr_decay: 1.0,
            patience: 0,
            min_delta: 0.0,
            restore_best: false,
        };
        let hist = model.train(&x, &y, &cfg);
        assert!(
            *hist.last().unwrap() < 0.1,
            "CCE final demasiado alta: {}",
            hist.last().unwrap()
        );

        // argmax de cada fila debe coincidir con la clase esperada.
        let pred = model.predict(&x);
        let expected = [0, 1, 2, 2];
        for (r, &want) in expected.iter().enumerate() {
            let mut best = 0;
            for c in 1..3 {
                if pred[[r, c]] > pred[[r, best]] {
                    best = c;
                }
            }
            assert_eq!(best, want, "fila {r}: predijo {best}, esperaba {want}");
        }

        // Cada fila de softmax debe sumar ~1.
        for r in 0..pred.nrows() {
            let s: f32 = (0..3).map(|c| pred[[r, c]]).sum();
            assert!((s - 1.0).abs() < 1e-4, "softmax fila {r} suma {s}");
        }
    }

    #[test]
    fn early_stopping_para_antes() {
        let mut rng = Rng::new(5);
        let (x, y) = xor_data();
        let mut model = xor_model(&mut rng);
        let mut cfg = TrainConfig::adam(5000, 0.05);
        cfg.loss = Loss::Bce;
        cfg.patience = 20;
        cfg.min_delta = 1e-4;
        let res = model.train_with_validation(&x, &y, None, &cfg);

        assert!(res.stopped_early, "debió parar por paciencia");
        assert!(res.history.len() < 5000, "no recortó épocas");
        assert_eq!(res.history.len(), res.best_epoch + cfg.patience + 1);
    }

    #[test]
    fn validacion_registra_val_history() {
        let mut rng = Rng::new(11);
        let (x, y) = xor_data();
        let mut model = xor_model(&mut rng);
        let mut cfg = TrainConfig::adam(200, 0.05);
        cfg.loss = Loss::Bce;
        // Usamos el mismo set como validación solo para verificar el cableado.
        let res = model.train_with_validation(&x, &y, Some((&x, &y)), &cfg);

        assert_eq!(res.val_history.len(), res.history.len());
        assert!(res.best_loss.is_finite());
        assert!(res.val_history.last().unwrap() < res.val_history.first().unwrap());
    }

    #[test]
    fn restore_best_devuelve_mejores_pesos() {
        let mut rng = Rng::new(21);
        let (x, y) = xor_data();
        let mut model = xor_model(&mut rng);
        let mut cfg = TrainConfig::adam(300, 0.05);
        cfg.loss = Loss::Bce;
        cfg.restore_best = true;
        let res = model.train_with_validation(&x, &y, Some((&x, &y)), &cfg);

        // Tras restaurar, la loss del modelo debe igualar la mejor observada.
        let now = model.evaluate(&x, &y, cfg.loss);
        assert!(
            (now - res.best_loss).abs() < 1e-5,
            "loss actual {now} != best {}",
            res.best_loss
        );
    }
}
