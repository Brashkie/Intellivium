//! Puente N-API: expone el motor de `neuroforge-core` a Node.js.
//!
//! Estrategia: el grafo de autograd vive ENTERO en Rust. Hacia JS solo cruzan
//! tensores planos (Float64Array + shape) y operaciones de alto nivel
//! (construir modelo, train, predict). Así no se marshalea el grafo por op,
//! que sería lento y frágil.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use ndarray::Array2;
use neuroforge_core::{Activation, Dense, Loss, Model, Optimizer, Rng, TrainConfig};

/// Especificación de una capa densa, recibida desde JS como objeto.
#[napi(object)]
pub struct LayerSpec {
    pub input_dim: u32,
    pub output_dim: u32,
    pub activation: String,
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
        }
    }
}

/// Estado serializable de una capa (para save/load desde JS).
#[napi(object)]
pub struct LayerState {
    pub input_dim: u32,
    pub output_dim: u32,
    pub activation: String,
    /// Pesos W aplanados row-major (input_dim * output_dim).
    pub weights: Float64Array,
    /// Bias (output_dim).
    pub bias: Float64Array,
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
            let act = Activation::from_str(&l.activation);
            built.push(Dense::new(
                l.input_dim as usize,
                l.output_dim as usize,
                act,
                &mut rng,
            ));
            out_dim = l.output_dim;
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
        self.inner
            .layers
            .iter()
            .map(|l| LayerState {
                input_dim: l.w.nrows() as u32,
                output_dim: l.w.ncols() as u32,
                activation: l.act.as_str().to_string(),
                weights: Float64Array::new(l.w.iter().map(|&v| v as f64).collect()),
                bias: Float64Array::new(l.b.iter().map(|&v| v as f64).collect()),
            })
            .collect()
    }

    /// Reemplaza los pesos de una capa (para cargar un modelo guardado).
    #[napi]
    pub fn set_weights(
        &mut self,
        index: u32,
        weights: Float64Array,
        bias: Float64Array,
    ) -> Result<()> {
        let i = index as usize;
        if i >= self.inner.layers.len() {
            return Err(Error::from_reason(format!("capa {i} fuera de rango")));
        }
        let (rows, cols) = {
            let w = &self.inner.layers[i].w;
            (w.nrows(), w.ncols())
        };
        let w = to_array2(&weights, rows, cols)?;
        let b = to_array2(&bias, 1, cols)?;
        self.inner.set_weights(i, w, b);
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
