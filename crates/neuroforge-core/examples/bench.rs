//! Micro-benchmark del bucle de entrenamiento (forward + backward + update).
//! Ejecutar con: `cargo run --release -p neuroforge-core --example bench`
use std::time::Instant;

use ndarray::Array2;
use neuroforge_core::{Activation, Layer, Loss, Model, Optimizer, Rng, TrainConfig};

fn main() {
    let mut rng = Rng::new(42);
    // MLP mediano: 64 -> 128 -> 128 -> 10
    let mut model = Model::new(vec![
        Layer::dense(64, 128, Activation::Relu, &mut rng),
        Layer::dense(128, 128, Activation::Relu, &mut rng),
        Layer::dense(128, 10, Activation::Softmax, &mut rng),
    ]);

    // Datos sintéticos: 256 muestras.
    let n = 256;
    let x = Array2::from_shape_fn((n, 64), |_| rng.normal());
    let y = Array2::from_shape_fn((n, 10), |(i, j)| if j == i % 10 { 1.0 } else { 0.0 });

    let mut cfg = TrainConfig {
        epochs: 50,
        lr: 0.01,
        loss: Loss::Cce,
        optimizer: Optimizer::adam_default(),
        batch_size: 64,
        grad_clip: 0.0,
        lr_decay: 1.0,
        patience: 0,
        min_delta: 0.0,
        restore_best: false,
    };

    // calentamiento
    cfg.epochs = 5;
    model.train(&x, &y, &cfg);

    cfg.epochs = 50;
    let t0 = Instant::now();
    let hist = model.train(&x, &y, &cfg);
    let dt = t0.elapsed();

    let steps = cfg.epochs * (n / cfg.batch_size);
    println!("épocas: {}  pasos: {}", cfg.epochs, steps);
    println!("tiempo total: {:.3?}", dt);
    println!("por paso:     {:.3?}", dt / steps as u32);
    println!("loss final:   {:.4}", hist.last().unwrap());
}
