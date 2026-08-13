//! Puente N-API: expone el motor de `neuroforge-core` a Node.js.
//!
//! Estrategia: el grafo de autograd vive ENTERO en Rust. Hacia JS solo cruzan
//! tensores planos (Float64Array + shape) y operaciones de alto nivel
//! (construir modelo, train, predict). Así no se marshalea el grafo por op,
//! que sería lento y frágil.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use ndarray::Array2;
use neuroforge_core::{Activation, Layer, Loss, Model, Optimizer, Rng, TrainConfig};

/// Especificación de una capa recibida desde JS.
/// `kind`: "dense" | "dropout" | "layernorm".
#[napi(object)]
pub struct LayerSpec {
    pub kind: Option<String>,
    pub input_dim: Option<u32>,
    pub output_dim: Option<u32>,
    pub activation: Option<String>,
    /// probabilidad de dropout
    pub p: Option<f64>,
    /// features de layernorm/batchnorm
    pub features: Option<u32>,
    /// embedding: tamaño del vocabulario
    pub vocab: Option<u32>,
    /// embedding: dimensión del vector
    pub dim: Option<u32>,
}

/// Config de entrenamiento recibida desde JS.
#[napi(object)]
pub struct JsTrainConfig {
    pub epochs: u32,
    pub lr: f64,
    /// "sgd" | "adam" (default: "sgd")
    pub optimizer: Option<String>,
    /// "mse" | "bce" (default: "mse")
    pub loss: Option<String>,
    pub beta1: Option<f64>,
    pub beta2: Option<f64>,
    pub eps: Option<f64>,
    /// Tamaño de mini-batch. 0/ausente = batch completo.
    pub batch_size: Option<u32>,
    /// Clipping de gradiente por norma L2 global. 0/ausente = desactivado.
    pub grad_clip: Option<f64>,
    /// Decaimiento del lr por época (lr * lr_decay^epoch). Ausente = 1.0.
    pub lr_decay: Option<f64>,
    /// Épocas sin mejora antes de parar. 0/ausente = desactivado.
    pub patience: Option<u32>,
    /// Mejora mínima para contar como progreso.
    pub min_delta: Option<f64>,
    /// Restaurar los pesos de la mejor época al terminar.
    pub restore_best: Option<bool>,
}

/// Resultado del entrenamiento devuelto a JS.
#[napi(object)]
pub struct TrainOutcome {
    pub history: Vec<f64>,
    pub val_history: Vec<f64>,
    pub best_epoch: u32,
    pub best_loss: f64,
    pub stopped_early: bool,
}

impl JsTrainConfig {
    fn to_core(&self) -> TrainConfig {
        let optimizer = match self.optimizer.as_deref() {
            Some("adam") => Optimizer::Adam {
                beta1: self.beta1.unwrap_or(0.9) as f32,
                beta2: self.beta2.unwrap_or(0.999) as f32,
                eps: self.eps.unwrap_or(1e-8) as f32,
            },
            _ => Optimizer::Sgd,
        };
        TrainConfig {
            epochs: self.epochs as usize,
            lr: self.lr as f32,
            loss: Loss::from_str(self.loss.as_deref().unwrap_or("mse")),
            optimizer,
            batch_size: self.batch_size.unwrap_or(0) as usize,
            grad_clip: self.grad_clip.unwrap_or(0.0) as f32,
            lr_decay: self.lr_decay.unwrap_or(1.0) as f32,
            patience: self.patience.unwrap_or(0) as usize,
            min_delta: self.min_delta.unwrap_or(0.0) as f32,
            restore_best: self.restore_best.unwrap_or(false),
        }
    }
}

/// Estado serializable de una capa (para save/load desde JS).
#[napi(object)]
pub struct LayerState {
    pub kind: String,
    pub input_dim: u32,
    pub output_dim: u32,
    pub activation: String,
    /// Pesos aplanados row-major (dense: W; layernorm: gamma). Vacío en dropout.
    pub weights: Float64Array,
    /// Bias / beta. Vacío en dropout.
    pub bias: Float64Array,
    pub p: f64,
    pub features: u32,
    /// BatchNorm: media móvil (running mean). Vacío en otras capas.
    pub running_mean: Float64Array,
    /// BatchNorm: varianza móvil (running var). Vacío en otras capas.
    pub running_var: Float64Array,
}

#[napi(js_name = "Model")]
pub struct JsModel {
    inner: Model,
    out_dim: u32,
}

#[napi]
impl JsModel {
    #[napi(constructor)]
    pub fn new(layers: Vec<LayerSpec>, seed: Option<f64>) -> Result<Self> {
        if layers.is_empty() {
            return Err(Error::from_reason("el modelo necesita al menos 1 capa"));
        }
        let mut rng = Rng::new(seed.unwrap_or(42.0) as u64);
        let mut built = Vec::with_capacity(layers.len());
        let mut out_dim = 0u32;
        for l in &layers {
            match l.kind.as_deref().unwrap_or("dense") {
                "dropout" => {
                    built.push(Layer::dropout(l.p.unwrap_or(0.5) as f32));
                }
                "layernorm" => {
                    let f = l
                        .features
                        .ok_or_else(|| Error::from_reason("layernorm requiere 'features'"))?;
                    built.push(Layer::layer_norm(f as usize));
                }
                "batchnorm" => {
                    let f = l
                        .features
                        .ok_or_else(|| Error::from_reason("batchnorm requiere 'features'"))?;
                    built.push(Layer::batch_norm(f as usize));
                }
                "embedding" => {
                    let v = l
                        .vocab
                        .ok_or_else(|| Error::from_reason("embedding requiere 'vocab'"))?;
                    let d = l
                        .dim
                        .ok_or_else(|| Error::from_reason("embedding requiere 'dim'"))?;
                    built.push(Layer::embedding(v as usize, d as usize, &mut rng));
                    out_dim = d;
                }
                _ => {
                    let inp = l
                        .input_dim
                        .ok_or_else(|| Error::from_reason("dense requiere 'inputDim'"))?;
                    let out = l
                        .output_dim
                        .ok_or_else(|| Error::from_reason("dense requiere 'outputDim'"))?;
                    let act = Activation::from_str(l.activation.as_deref().unwrap_or("linear"));
                    built.push(Layer::dense(inp as usize, out as usize, act, &mut rng));
                    out_dim = out;
                }
            }
        }
        Ok(JsModel {
            inner: Model::new(built),
            out_dim,
        })
    }

    /// Entrena según la config (optimizer + loss). Devuelve el historial de loss.
    #[napi]
    #[allow(clippy::too_many_arguments)]
    pub fn train(
        &mut self,
        x: Float64Array,
        x_rows: u32,
        x_cols: u32,
        y: Float64Array,
        y_rows: u32,
        y_cols: u32,
        config: JsTrainConfig,
    ) -> Result<Vec<f64>> {
        let xm = to_array2(&x, x_rows as usize, x_cols as usize)?;
        let ym = to_array2(&y, y_rows as usize, y_cols as usize)?;
        let hist = self.inner.train(&xm, &ym, &config.to_core());
        Ok(hist.into_iter().map(|v| v as f64).collect())
    }

    /// Entrena con validación opcional, early stopping y checkpoint del mejor
    /// modelo. Devuelve historiales + metadatos.
    #[napi]
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        &mut self,
        x: Float64Array,
        x_rows: u32,
        x_cols: u32,
        y: Float64Array,
        y_rows: u32,
        y_cols: u32,
        config: JsTrainConfig,
        val_x: Option<Float64Array>,
        val_x_rows: Option<u32>,
        val_y: Option<Float64Array>,
        val_y_cols: Option<u32>,
    ) -> Result<TrainOutcome> {
        let xm = to_array2(&x, x_rows as usize, x_cols as usize)?;
        let ym = to_array2(&y, y_rows as usize, y_cols as usize)?;

        let val_pair = match (val_x, val_x_rows, val_y, val_y_cols) {
            (Some(vx), Some(vr), Some(vy), Some(vc)) => {
                let vxm = to_array2(&vx, vr as usize, x_cols as usize)?;
                let vym = to_array2(&vy, vr as usize, vc as usize)?;
                Some((vxm, vym))
            }
            _ => None,
        };

        let res = self.inner.train_with_validation(
            &xm,
            &ym,
            val_pair.as_ref().map(|(a, b)| (a, b)),
            &config.to_core(),
        );

        Ok(TrainOutcome {
            history: res.history.into_iter().map(|v| v as f64).collect(),
            val_history: res.val_history.into_iter().map(|v| v as f64).collect(),
            best_epoch: res.best_epoch as u32,
            best_loss: res.best_loss as f64,
            stopped_early: res.stopped_early,
        })
    }

    /// Calcula la loss sobre un conjunto, sin entrenar.
    #[napi]
    pub fn evaluate(
        &self,
        x: Float64Array,
        x_rows: u32,
        x_cols: u32,
        y: Float64Array,
        y_cols: u32,
        loss: Option<String>,
    ) -> Result<f64> {
        let xm = to_array2(&x, x_rows as usize, x_cols as usize)?;
        let ym = to_array2(&y, x_rows as usize, y_cols as usize)?;
        let l = Loss::from_str(loss.as_deref().unwrap_or("mse"));
        Ok(self.inner.evaluate(&xm, &ym, l) as f64)
    }

    /// Predice. Devuelve un Float64Array plano (row-major) de shape (x_rows, out_dim).
    #[napi]
    pub fn predict(&self, x: Float64Array, x_rows: u32, x_cols: u32) -> Result<Float64Array> {
        let xm = to_array2(&x, x_rows as usize, x_cols as usize)?;
        let out = self.inner.predict(&xm);
        let flat: Vec<f64> = out.iter().map(|&v| v as f64).collect();
        Ok(Float64Array::new(flat))
    }

    #[napi(getter)]
    pub fn output_dim(&self) -> u32 {
        self.out_dim
    }

    /// Serializa los pesos de todas las capas (para guardar el modelo).
    #[napi]
    pub fn save(&self) -> Vec<LayerState> {
        let empty = || Float64Array::new(vec![]);
        let flat = |a: &Array2<f32>| Float64Array::new(a.iter().map(|&v| v as f64).collect());
        (0..self.inner.layer_count())
            .map(|i| {
                if let Some(d) = self.inner.dense_at(i) {
                    LayerState {
                        kind: "dense".to_string(),
                        input_dim: d.w.nrows() as u32,
                        output_dim: d.w.ncols() as u32,
                        activation: d.act.as_str().to_string(),
                        weights: flat(&d.w),
                        bias: flat(&d.b),
                        p: 0.0,
                        features: 0,
                        running_mean: empty(),
                        running_var: empty(),
                    }
                } else if let Some(ln) = self.inner.layernorm_at(i) {
                    LayerState {
                        kind: "layernorm".to_string(),
                        input_dim: 0,
                        output_dim: 0,
                        activation: "linear".to_string(),
                        weights: flat(&ln.gamma),
                        bias: flat(&ln.beta),
                        p: 0.0,
                        features: ln.gamma.ncols() as u32,
                        running_mean: empty(),
                        running_var: empty(),
                    }
                } else if let Some(bn) = self.inner.batchnorm_at(i) {
                    LayerState {
                        kind: "batchnorm".to_string(),
                        input_dim: 0,
                        output_dim: 0,
                        activation: "linear".to_string(),
                        weights: flat(&bn.gamma),
                        bias: flat(&bn.beta),
                        p: 0.0,
                        features: bn.gamma.ncols() as u32,
                        running_mean: flat(&bn.running_mean),
                        running_var: flat(&bn.running_var),
                    }
                } else if let Some(emb) = self.inner.embedding_at(i) {
                    LayerState {
                        kind: "embedding".to_string(),
                        input_dim: emb.table.nrows() as u32, // vocab
                        output_dim: emb.table.ncols() as u32, // dim
                        activation: "linear".to_string(),
                        weights: flat(&emb.table),
                        bias: empty(),
                        p: 0.0,
                        features: 0,
                        running_mean: empty(),
                        running_var: empty(),
                    }
                } else {
                    LayerState {
                        kind: "dropout".to_string(),
                        input_dim: 0,
                        output_dim: 0,
                        activation: "linear".to_string(),
                        weights: empty(),
                        bias: empty(),
                        p: self.inner.dropout_p(i).unwrap_or(0.0) as f64,
                        features: 0,
                        running_mean: empty(),
                        running_var: empty(),
                    }
                }
            })
            .collect()
    }

    /// Reemplaza los pesos de una capa (dense o layernorm) al cargar un modelo.
    #[napi]
    pub fn set_weights(
        &mut self,
        index: u32,
        weights: Float64Array,
        bias: Float64Array,
    ) -> Result<()> {
        let i = index as usize;
        if i >= self.inner.layer_count() {
            return Err(Error::from_reason(format!("capa {i} fuera de rango")));
        }
        if let Some((rows, cols)) = self.inner.dense_at(i).map(|d| (d.w.nrows(), d.w.ncols())) {
            let w = to_array2(&weights, rows, cols)?;
            let b = to_array2(&bias, 1, cols)?;
            self.inner.set_dense_weights(i, w, b);
        } else if let Some(feats) = self.inner.layernorm_at(i).map(|ln| ln.gamma.ncols()) {
            let g = to_array2(&weights, 1, feats)?;
            let b = to_array2(&bias, 1, feats)?;
            self.inner.set_layernorm_weights(i, g, b);
        }
        Ok(())
    }

    /// Reemplaza gamma/beta y las running stats de una capa BatchNorm (para load).
    #[napi]
    pub fn set_batchnorm_weights(
        &mut self,
        index: u32,
        gamma: Float64Array,
        beta: Float64Array,
        running_mean: Float64Array,
        running_var: Float64Array,
    ) -> Result<()> {
        let i = index as usize;
        let feats = match self.inner.batchnorm_at(i).map(|bn| bn.gamma.ncols()) {
            Some(f) => f,
            None => return Ok(()),
        };
        let g = to_array2(&gamma, 1, feats)?;
        let b = to_array2(&beta, 1, feats)?;
        let rm = to_array2(&running_mean, 1, feats)?;
        let rv = to_array2(&running_var, 1, feats)?;
        self.inner.set_batchnorm_weights(i, g, b, rm, rv);
        Ok(())
    }

    /// Reemplaza la tabla de una capa Embedding (para load). `table` aplanada
    /// row-major (vocab * dim).
    #[napi]
    pub fn set_embedding_table(
        &mut self,
        index: u32,
        vocab: u32,
        dim: u32,
        table: Float64Array,
    ) -> Result<()> {
        let i = index as usize;
        if self.inner.embedding_at(i).is_none() {
            return Ok(());
        }
        let t = to_array2(&table, vocab as usize, dim as usize)?;
        self.inner.set_embedding_table(i, t);
        Ok(())
    }
}

fn to_array2(data: &Float64Array, rows: usize, cols: usize) -> Result<Array2<f32>> {
    let slice = data.as_ref();
    if slice.len() != rows * cols {
        return Err(Error::from_reason(format!(
            "shape inválido: len={} pero rows*cols={}",
            slice.len(),
            rows * cols
        )));
    }
    let v: Vec<f32> = slice.iter().map(|&x| x as f32).collect();
    Array2::from_shape_vec((rows, cols), v).map_err(|e| Error::from_reason(e.to_string()))
}
