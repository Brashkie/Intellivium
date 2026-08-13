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
    LeakyRelu,
    Elu,
    Gelu,
}

impl Activation {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Activation {
        match s.to_lowercase().as_str() {
            "relu" => Activation::Relu,
            "sigmoid" => Activation::Sigmoid,
            "tanh" => Activation::Tanh,
            "softmax" => Activation::Softmax,
            "leakyrelu" | "leaky_relu" | "leaky" => Activation::LeakyRelu,
            "elu" => Activation::Elu,
            "gelu" => Activation::Gelu,
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
            Activation::LeakyRelu => "leakyrelu",
            Activation::Elu => "elu",
            Activation::Gelu => "gelu",
        }
    }
}

/// Función de pérdida.
#[derive(Clone, Copy, Debug)]
pub enum Loss {
    Mse,
    Bce,
    Cce,
    Mae,
    Huber,
}

impl Loss {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Loss {
        match s.to_lowercase().as_str() {
            "bce" | "binary_crossentropy" => Loss::Bce,
            "cce" | "categorical_crossentropy" | "crossentropy" => Loss::Cce,
            "mae" | "l1" => Loss::Mae,
            "huber" | "smooth_l1" => Loss::Huber,
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
    mw: Array2<f32>,
    vw: Array2<f32>,
    mb: Array2<f32>,
    vb: Array2<f32>,
}

impl Dense {
    pub fn new(inp: usize, out: usize, act: Activation, rng: &mut Rng) -> Self {
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
}

/// Normalización por muestra (Layer Normalization) sobre las columnas.
pub struct LayerNorm {
    pub gamma: Array2<f32>, // (1, features)
    pub beta: Array2<f32>,  // (1, features)
    pub eps: f32,
    mg: Array2<f32>,
    vg: Array2<f32>,
    mb: Array2<f32>,
    vb: Array2<f32>,
}

impl LayerNorm {
    pub fn new(features: usize) -> Self {
        LayerNorm {
            gamma: Array2::ones((1, features)),
            beta: Array2::zeros((1, features)),
            eps: 1e-5,
            mg: Array2::zeros((1, features)),
            vg: Array2::zeros((1, features)),
            mb: Array2::zeros((1, features)),
            vb: Array2::zeros((1, features)),
        }
    }
}

/// Batch Normalization por columnas con estadísticas móviles (running stats).
/// En training normaliza con stats del batch y actualiza las running; en eval
/// (predict/evaluate) normaliza con las running.
pub struct BatchNorm {
    pub gamma: Array2<f32>,        // (1, features)
    pub beta: Array2<f32>,         // (1, features)
    pub running_mean: Array2<f32>, // (1, features)
    pub running_var: Array2<f32>,  // (1, features)
    pub momentum: f32,
    pub eps: f32,
    mg: Array2<f32>,
    vg: Array2<f32>,
    mb: Array2<f32>,
    vb: Array2<f32>,
}

impl BatchNorm {
    pub fn new(features: usize) -> Self {
        BatchNorm {
            gamma: Array2::ones((1, features)),
            beta: Array2::zeros((1, features)),
            running_mean: Array2::zeros((1, features)),
            running_var: Array2::ones((1, features)),
            momentum: 0.1,
            eps: 1e-5,
            mg: Array2::zeros((1, features)),
            vg: Array2::zeros((1, features)),
            mb: Array2::zeros((1, features)),
            vb: Array2::zeros((1, features)),
        }
    }
}

/// Capa de Embedding: tabla entrenable (vocab, dim). Mapea índices enteros
/// (pasados como f32) a vectores densos. Entrada (batch, L) -> salida (batch, L*dim).
pub struct Embedding {
    pub table: Array2<f32>, // (vocab, dim)
    mt: Array2<f32>,
    vt: Array2<f32>,
}

impl Embedding {
    pub fn new(vocab: usize, dim: usize, rng: &mut Rng) -> Self {
        let table = Array2::from_shape_fn((vocab, dim), |_| rng.normal() * 0.1);
        Embedding {
            table,
            mt: Array2::zeros((vocab, dim)),
            vt: Array2::zeros((vocab, dim)),
        }
    }

    pub fn dim(&self) -> usize {
        self.table.ncols()
    }
}

/// Una capa del modelo secuencial.
pub enum Layer {
    Dense(Dense),
    /// Dropout con probabilidad p (inverted dropout; activo solo en training).
    Dropout(f32),
    LayerNorm(LayerNorm),
    BatchNorm(BatchNorm),
    Embedding(Embedding),
}

impl Layer {
    pub fn dense(inp: usize, out: usize, act: Activation, rng: &mut Rng) -> Layer {
        Layer::Dense(Dense::new(inp, out, act, rng))
    }
    pub fn dropout(p: f32) -> Layer {
        Layer::Dropout(p)
    }
    pub fn layer_norm(features: usize) -> Layer {
        Layer::LayerNorm(LayerNorm::new(features))
    }
    pub fn batch_norm(features: usize) -> Layer {
        Layer::BatchNorm(BatchNorm::new(features))
    }
    pub fn embedding(vocab: usize, dim: usize, rng: &mut Rng) -> Layer {
        Layer::Embedding(Embedding::new(vocab, dim, rng))
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Layer::Dense(_) => "dense",
            Layer::Dropout(_) => "dropout",
            Layer::LayerNorm(_) => "layernorm",
            Layer::BatchNorm(_) => "batchnorm",
            Layer::Embedding(_) => "embedding",
        }
    }

    /// Aplica los gradientes a los parámetros de la capa. `ids` son los índices
    /// de la cinta de los parámetros en el orden en que se registraron.
    fn apply_grads(
        &mut self,
        ids: &[usize],
        grads: &[Array2<f32>],
        opt: &Optimizer,
        lr: f32,
        t: i32,
        scale: f32,
    ) {
        match self {
            Layer::Dense(d) => {
                let gw = &grads[ids[0]] * scale;
                let gb = &grads[ids[1]] * scale;
                apply_param(&mut d.w, &mut d.mw, &mut d.vw, &gw, opt, lr, t);
                apply_param(&mut d.b, &mut d.mb, &mut d.vb, &gb, opt, lr, t);
            }
            Layer::Dropout(_) => {}
            Layer::LayerNorm(ln) => {
                let gg = &grads[ids[0]] * scale;
                let gb = &grads[ids[1]] * scale;
                apply_param(&mut ln.gamma, &mut ln.mg, &mut ln.vg, &gg, opt, lr, t);
                apply_param(&mut ln.beta, &mut ln.mb, &mut ln.vb, &gb, opt, lr, t);
            }
            Layer::BatchNorm(bn) => {
                let gg = &grads[ids[0]] * scale;
                let gb = &grads[ids[1]] * scale;
                apply_param(&mut bn.gamma, &mut bn.mg, &mut bn.vg, &gg, opt, lr, t);
                apply_param(&mut bn.beta, &mut bn.mb, &mut bn.vb, &gb, opt, lr, t);
            }
            Layer::Embedding(emb) => {
                let gt = &grads[ids[0]] * scale;
                apply_param(&mut emb.table, &mut emb.mt, &mut emb.vt, &gt, opt, lr, t);
            }
        }
    }
}

/// Media y varianza (sesgada) por columna sobre las filas (batch).
/// Devuelve dos arrays (1, cols).
fn column_stats(x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
    let (rows, cols) = (x.nrows(), x.ncols());
    let nf = rows as f32;
    let mut mean = Array2::<f32>::zeros((1, cols));
    let mut var = Array2::<f32>::zeros((1, cols));
    for c in 0..cols {
        let mut m = 0.0;
        for r in 0..rows {
            m += x[[r, c]];
        }
        m /= nf;
        let mut v = 0.0;
        for r in 0..rows {
            let d = x[[r, c]] - m;
            v += d * d;
        }
        v /= nf;
        mean[[0, c]] = m;
        var[[0, c]] = v;
    }
    (mean, var)
}

/// Actualiza un parámetro con SGD o Adam.
fn apply_param(
    p: &mut Array2<f32>,
    m: &mut Array2<f32>,
    v: &mut Array2<f32>,
    g: &Array2<f32>,
    opt: &Optimizer,
    lr: f32,
    t: i32,
) {
    match *opt {
        Optimizer::Sgd => {
            *p = &*p - &(g * lr);
        }
        Optimizer::Adam { beta1, beta2, eps } => {
            adam_step(p, m, v, g, lr, beta1, beta2, eps, t);
        }
    }
}

pub struct Model {
    pub layers: Vec<Layer>,
    t: i32,
    rng: Rng,
}

impl Model {
    pub fn new(layers: Vec<Layer>) -> Self {
        Model {
            layers,
            t: 0,
            rng: Rng::new(0x1234_5678),
        }
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn dense_at(&self, i: usize) -> Option<&Dense> {
        match &self.layers[i] {
            Layer::Dense(d) => Some(d),
            _ => None,
        }
    }

    pub fn layernorm_at(&self, i: usize) -> Option<&LayerNorm> {
        match &self.layers[i] {
            Layer::LayerNorm(ln) => Some(ln),
            _ => None,
        }
    }

    pub fn dropout_p(&self, i: usize) -> Option<f32> {
        match &self.layers[i] {
            Layer::Dropout(p) => Some(*p),
            _ => None,
        }
    }

    pub fn batchnorm_at(&self, i: usize) -> Option<&BatchNorm> {
        match &self.layers[i] {
            Layer::BatchNorm(bn) => Some(bn),
            _ => None,
        }
    }

    pub fn embedding_at(&self, i: usize) -> Option<&Embedding> {
        match &self.layers[i] {
            Layer::Embedding(e) => Some(e),
            _ => None,
        }
    }

    /// Reemplaza la tabla de una capa Embedding (para load).
    pub fn set_embedding_table(&mut self, i: usize, table: Array2<f32>) {
        if let Layer::Embedding(e) = &mut self.layers[i] {
            e.mt = Array2::zeros(table.raw_dim());
            e.vt = Array2::zeros(table.raw_dim());
            e.table = table;
        }
    }

    /// Reemplaza W y b de una capa densa (para load).
    pub fn set_dense_weights(&mut self, i: usize, w: Array2<f32>, b: Array2<f32>) {
        if let Layer::Dense(d) = &mut self.layers[i] {
            d.mw = Array2::zeros(w.raw_dim());
            d.vw = Array2::zeros(w.raw_dim());
            d.mb = Array2::zeros(b.raw_dim());
            d.vb = Array2::zeros(b.raw_dim());
            d.w = w;
            d.b = b;
        }
    }

    /// Reemplaza gamma y beta de una capa LayerNorm (para load).
    pub fn set_layernorm_weights(&mut self, i: usize, gamma: Array2<f32>, beta: Array2<f32>) {
        if let Layer::LayerNorm(ln) = &mut self.layers[i] {
            ln.mg = Array2::zeros(gamma.raw_dim());
            ln.vg = Array2::zeros(gamma.raw_dim());
            ln.mb = Array2::zeros(beta.raw_dim());
            ln.vb = Array2::zeros(beta.raw_dim());
            ln.gamma = gamma;
            ln.beta = beta;
        }
    }

    /// Reemplaza gamma/beta y las running stats de una capa BatchNorm (para load).
    pub fn set_batchnorm_weights(
        &mut self,
        i: usize,
        gamma: Array2<f32>,
        beta: Array2<f32>,
        running_mean: Array2<f32>,
        running_var: Array2<f32>,
    ) {
        if let Layer::BatchNorm(bn) = &mut self.layers[i] {
            bn.mg = Array2::zeros(gamma.raw_dim());
            bn.vg = Array2::zeros(gamma.raw_dim());
            bn.mb = Array2::zeros(beta.raw_dim());
            bn.vb = Array2::zeros(beta.raw_dim());
            bn.gamma = gamma;
            bn.beta = beta;
            bn.running_mean = running_mean;
            bn.running_var = running_var;
        }
    }

    /// Construye el grafo forward. `training` activa Dropout y hace que BatchNorm
    /// use stats del batch. Devuelve (salida, ids de parámetros por capa,
    /// actualizaciones de running stats de BatchNorm: (índice de capa, mean, var)).
    #[allow(clippy::type_complexity)]
    fn forward_tape(
        &self,
        tape: &mut Tape,
        x: usize,
        training: bool,
        mut rng: Option<&mut Rng>,
    ) -> (
        usize,
        Vec<Vec<usize>>,
        Vec<(usize, Array2<f32>, Array2<f32>)>,
    ) {
        let mut cur = x;
        let mut params: Vec<Vec<usize>> = Vec::with_capacity(self.layers.len());
        let mut bn_updates: Vec<(usize, Array2<f32>, Array2<f32>)> = Vec::new();
        for (li, layer) in self.layers.iter().enumerate() {
            match layer {
                Layer::Dense(d) => {
                    let wid = tape.leaf(d.w.clone());
                    let bid = tape.leaf(d.b.clone());
                    let z = tape.matmul(cur, wid);
                    let z = tape.add(z, bid);
                    cur = match d.act {
                        Activation::Linear => z,
                        Activation::Relu => tape.relu(z),
                        Activation::Sigmoid => tape.sigmoid(z),
                        Activation::Tanh => tape.tanh(z),
                        Activation::Softmax => tape.softmax(z),
                        Activation::LeakyRelu => tape.leaky_relu(z),
                        Activation::Elu => tape.elu(z),
                        Activation::Gelu => tape.gelu(z),
                    };
                    params.push(vec![wid, bid]);
                }
                Layer::Dropout(p) => {
                    if training && *p > 0.0 {
                        if let Some(r) = rng.as_deref_mut() {
                            cur = tape.dropout(cur, *p, r);
                        }
                    }
                    params.push(vec![]);
                }
                Layer::LayerNorm(ln) => {
                    let gid = tape.leaf(ln.gamma.clone());
                    let bid = tape.leaf(ln.beta.clone());
                    cur = tape.layer_norm(cur, gid, bid, ln.eps);
                    params.push(vec![gid, bid]);
                }
                Layer::BatchNorm(bn) => {
                    let gid = tape.leaf(bn.gamma.clone());
                    let bid = tape.leaf(bn.beta.clone());
                    if training {
                        let (mean, var) = column_stats(tape.value(cur));
                        cur = tape.batch_norm(cur, gid, bid, &mean, &var, bn.eps);
                        bn_updates.push((li, mean, var));
                    } else {
                        cur = tape.batch_norm(
                            cur,
                            gid,
                            bid,
                            &bn.running_mean,
                            &bn.running_var,
                            bn.eps,
                        );
                    }
                    params.push(vec![gid, bid]);
                }
                Layer::Embedding(emb) => {
                    let tid = tape.leaf(emb.table.clone());
                    cur = tape.embedding(cur, tid, emb.dim());
                    params.push(vec![tid]);
                }
            }
        }
        (cur, params, bn_updates)
    }

    pub fn predict(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut tape = Tape::new();
        let xid = tape.leaf(x.clone());
        let (out, _, _) = self.forward_tape(&mut tape, xid, false, None);
        tape.value(out).clone()
    }

    fn build_loss(tape: &mut Tape, loss: Loss, out: usize, yid: usize) -> usize {
        match loss {
            Loss::Mse => tape.mse(out, yid),
            Loss::Bce => tape.bce(out, yid),
            Loss::Cce => tape.cce(out, yid),
            Loss::Mae => tape.mae(out, yid),
            Loss::Huber => tape.huber(out, yid),
        }
    }

    /// Un paso de entrenamiento sobre un batch. Aplica clipping global si procede.
    fn step(&mut self, xb: &Array2<f32>, yb: &Array2<f32>, cfg: &TrainConfig, lr: f32) -> f32 {
        let mut tape = Tape::new();
        let xid = tape.leaf(xb.clone());
        let yid = tape.leaf(yb.clone());

        let mut rng = std::mem::replace(&mut self.rng, Rng::new(1));
        let (out, param_ids, bn_updates) = self.forward_tape(&mut tape, xid, true, Some(&mut rng));
        self.rng = rng;

        let loss = Self::build_loss(&mut tape, cfg.loss, out, yid);
        let loss_val = tape.value(loss)[[0, 0]];
        let grads = tape.backward(loss);

        // Actualiza las running stats de cada BatchNorm con las del batch (EMA).
        for (li, mean, var) in &bn_updates {
            if let Layer::BatchNorm(bn) = &mut self.layers[*li] {
                let m = bn.momentum;
                bn.running_mean = &(&bn.running_mean * (1.0 - m)) + &(mean * m);
                bn.running_var = &(&bn.running_var * (1.0 - m)) + &(var * m);
            }
        }

        // Clipping por norma L2 global sobre todos los parámetros.
        let mut scale = 1.0f32;
        if cfg.grad_clip > 0.0 {
            let mut sq = 0.0f32;
            for ids in &param_ids {
                for &id in ids {
                    sq += grads[id].iter().map(|&v| v * v).sum::<f32>();
                }
            }
            let norm = sq.sqrt();
            if norm > cfg.grad_clip {
                scale = cfg.grad_clip / (norm + 1e-12);
            }
        }

        self.t += 1;
        for (li, ids) in param_ids.iter().enumerate() {
            if !ids.is_empty() {
                self.layers[li].apply_grads(ids, &grads, &cfg.optimizer, lr, self.t, scale);
            }
        }
        loss_val
    }

    /// Calcula la loss sobre un conjunto sin actualizar pesos (modo eval).
    pub fn evaluate(&self, x: &Array2<f32>, y: &Array2<f32>, loss: Loss) -> f32 {
        let mut tape = Tape::new();
        let xid = tape.leaf(x.clone());
        let yid = tape.leaf(y.clone());
        let (out, _, _) = self.forward_tape(&mut tape, xid, false, None);
        let l = Self::build_loss(&mut tape, loss, out, yid);
        tape.value(l)[[0, 0]]
    }

    /// Copia todos los parámetros entrenables (checkpoint en memoria).
    fn snapshot(&self) -> Vec<Vec<Array2<f32>>> {
        self.layers
            .iter()
            .map(|l| match l {
                Layer::Dense(d) => vec![d.w.clone(), d.b.clone()],
                Layer::Dropout(_) => vec![],
                Layer::LayerNorm(ln) => vec![ln.gamma.clone(), ln.beta.clone()],
                Layer::BatchNorm(bn) => vec![
                    bn.gamma.clone(),
                    bn.beta.clone(),
                    bn.running_mean.clone(),
                    bn.running_var.clone(),
                ],
                Layer::Embedding(e) => vec![e.table.clone()],
            })
            .collect()
    }

    fn restore(&mut self, snap: Vec<Vec<Array2<f32>>>) {
        for (i, params) in snap.into_iter().enumerate() {
            match &mut self.layers[i] {
                Layer::Dense(d) => {
                    d.w = params[0].clone();
                    d.b = params[1].clone();
                }
                Layer::Dropout(_) => {}
                Layer::LayerNorm(ln) => {
                    ln.gamma = params[0].clone();
                    ln.beta = params[1].clone();
                }
                Layer::BatchNorm(bn) => {
                    bn.gamma = params[0].clone();
                    bn.beta = params[1].clone();
                    bn.running_mean = params[2].clone();
                    bn.running_var = params[3].clone();
                }
                Layer::Embedding(e) => {
                    e.table = params[0].clone();
                }
            }
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
        let mut best_snap: Option<Vec<Vec<Array2<f32>>>> = None;
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

    /// Entrena según `cfg`. Devuelve la loss media por época.
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
            Layer::dense(2, 8, Activation::Tanh, rng),
            Layer::dense(8, 1, Activation::Sigmoid, rng),
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
        for i in 0..trained.layer_count() {
            let d = trained.dense_at(i).unwrap();
            restored.set_dense_weights(i, d.w.clone(), d.b.clone());
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
            Layer::dense(2, 12, Activation::Relu, &mut rng),
            Layer::dense(12, 3, Activation::Softmax, &mut rng),
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

    #[test]
    fn nuevas_activaciones_y_parsers() {
        assert!(matches!(
            Activation::from_str("leakyrelu"),
            Activation::LeakyRelu
        ));
        assert!(matches!(Activation::from_str("elu"), Activation::Elu));
        assert!(matches!(Activation::from_str("gelu"), Activation::Gelu));
        assert!(matches!(Loss::from_str("mae"), Loss::Mae));
        assert!(matches!(Loss::from_str("huber"), Loss::Huber));

        // round-trip as_str/from_str
        for a in [Activation::LeakyRelu, Activation::Elu, Activation::Gelu] {
            assert!(matches!(
                Activation::from_str(a.as_str()),
                x if std::mem::discriminant(&x) == std::mem::discriminant(&a)
            ));
        }
    }

    #[test]
    fn learns_xor_gelu_huber() {
        let mut rng = Rng::new(13);
        let (x, y) = xor_data();
        let mut model = Model::new(vec![
            Layer::dense(2, 8, Activation::Gelu, &mut rng),
            Layer::dense(8, 1, Activation::Sigmoid, &mut rng),
        ]);
        let mut cfg = TrainConfig::adam(3000, 0.03);
        cfg.loss = Loss::Huber;
        let hist = model.train(&x, &y, &cfg);
        assert!(
            hist.last().unwrap() < hist.first().unwrap(),
            "Huber no bajó"
        );
        assert_xor(&model, &x);
    }

    #[test]
    fn mae_baja_con_leakyrelu() {
        let mut rng = Rng::new(4);
        let x = array![[0.0], [1.0], [2.0], [3.0]];
        let y = array![[0.0], [2.0], [4.0], [6.0]]; // y = 2x
        let mut model = Model::new(vec![
            Layer::dense(1, 8, Activation::LeakyRelu, &mut rng),
            Layer::dense(8, 1, Activation::Linear, &mut rng),
        ]);
        let mut cfg = TrainConfig::adam(2000, 0.02);
        cfg.loss = Loss::Mae;
        let hist = model.train(&x, &y, &cfg);
        assert!(
            *hist.last().unwrap() < 0.3,
            "MAE final: {}",
            hist.last().unwrap()
        );
    }

    #[test]
    fn dropout_es_identidad_en_eval() {
        let mut rng = Rng::new(2);
        let mut model = Model::new(vec![
            Layer::dense(2, 6, Activation::Relu, &mut rng),
            Layer::dropout(0.5),
            Layer::dense(6, 1, Activation::Linear, &mut rng),
        ]);
        let (x, _) = xor_data();
        let a = model.predict(&x);
        let b = model.predict(&x);
        for r in 0..a.nrows() {
            assert!((a[[r, 0]] - b[[r, 0]]).abs() < 1e-6);
        }
        let mut cfg = TrainConfig::adam(300, 0.05);
        cfg.loss = Loss::Mse;
        let y = array![[0.0], [1.0], [1.0], [0.0]];
        let hist = model.train(&x, &y, &cfg);
        assert!(hist.last().unwrap().is_finite());
    }

    #[test]
    fn layernorm_normaliza_y_entrena() {
        let mut rng = Rng::new(8);
        let ln_only = Model::new(vec![
            Layer::dense(3, 4, Activation::Linear, &mut rng),
            Layer::layer_norm(4),
        ]);
        let x = array![[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]];
        let out = ln_only.predict(&x);
        for r in 0..out.nrows() {
            let mean: f32 = (0..out.ncols()).map(|c| out[[r, c]]).sum::<f32>() / out.ncols() as f32;
            let var: f32 = (0..out.ncols())
                .map(|c| (out[[r, c]] - mean).powi(2))
                .sum::<f32>()
                / out.ncols() as f32;
            assert!(mean.abs() < 1e-3, "media fila {r} = {mean}");
            assert!((var - 1.0).abs() < 1e-2, "var fila {r} = {var}");
        }

        let mut m2 = Model::new(vec![
            Layer::dense(2, 8, Activation::Relu, &mut rng),
            Layer::layer_norm(8),
            Layer::dense(8, 1, Activation::Sigmoid, &mut rng),
        ]);
        let (xx, yy) = xor_data();
        let mut cfg = TrainConfig::adam(3000, 0.03);
        cfg.loss = Loss::Bce;
        let hist = m2.train(&xx, &yy, &cfg);
        assert!(
            hist.last().unwrap() < hist.first().unwrap(),
            "LayerNorm no bajó la loss"
        );
    }

    #[test]
    fn batchnorm_entrena_y_actualiza_running() {
        let mut rng = Rng::new(15);
        let mut model = Model::new(vec![
            Layer::dense(2, 8, Activation::Relu, &mut rng),
            Layer::batch_norm(8),
            Layer::dense(8, 1, Activation::Sigmoid, &mut rng),
        ]);
        let (x, y) = xor_data();

        // running stats arrancan en mean=0, var=1
        let (m0, v0) = {
            let bn = model.batchnorm_at(1).unwrap();
            (bn.running_mean.clone(), bn.running_var.clone())
        };
        assert!(m0.iter().all(|&v| v == 0.0));
        assert!(v0.iter().all(|&v| v == 1.0));

        let mut cfg = TrainConfig::adam(2000, 0.03);
        cfg.loss = Loss::Bce;
        let hist = model.train(&x, &y, &cfg);
        assert!(
            hist.last().unwrap() < hist.first().unwrap(),
            "BatchNorm no bajó la loss"
        );

        // tras entrenar, las running stats ya no son las iniciales
        let bn = model.batchnorm_at(1).unwrap();
        let cambio_mean = bn
            .running_mean
            .iter()
            .zip(m0.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        let cambio_var = bn
            .running_var
            .iter()
            .zip(v0.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            cambio_mean && cambio_var,
            "running stats no se actualizaron"
        );

        // predict (eval) es determinista y finito
        let p = model.predict(&x);
        assert!(p.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn embedding_aprende_por_indice() {
        // 4 índices (0..3), cada uno debe mapear a una salida distinta.
        let mut rng = Rng::new(9);
        let mut model = Model::new(vec![
            Layer::embedding(4, 8, &mut rng), // (batch,1) -> (batch,8)
            Layer::dense(8, 1, Activation::Sigmoid, &mut rng),
        ]);
        // índices como f32, columna única
        let x = array![[0.0], [1.0], [2.0], [3.0]];
        let y = array![[0.0], [1.0], [1.0], [0.0]]; // patrón arbitrario por índice
        let mut cfg = TrainConfig::adam(1500, 0.05);
        cfg.loss = Loss::Bce;
        let hist = model.train(&x, &y, &cfg);
        assert!(
            *hist.last().unwrap() < 0.05,
            "Embedding no aprendió: {}",
            hist.last().unwrap()
        );

        // salida (batch, 1) y predicciones correctas por índice
        let pred = model.predict(&x);
        assert_eq!(pred.ncols(), 1);
        assert!(pred[[0, 0]] < 0.5 && pred[[3, 0]] < 0.5);
        assert!(pred[[1, 0]] > 0.5 && pred[[2, 0]] > 0.5);
    }
}
