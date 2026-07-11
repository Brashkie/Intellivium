//! Prueba real del motor: aprender XOR (problema no lineal clásico).
//! Ahora con Adam + BCE. Si converge, el autograd + optimizador están bien.

use ndarray::array;
use neuroforge_core::{Activation, Dense, Loss, Model, Optimizer, Rng, TrainConfig};

fn main() {
    let mut rng = Rng::new(7);

    let x = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let y = array![[0.0], [1.0], [1.0], [0.0]];

    let mut model = Model::new(vec![
        Dense::new(2, 8, Activation::Tanh, &mut rng),
        Dense::new(8, 1, Activation::Sigmoid, &mut rng),
    ]);

    let cfg = TrainConfig {
        epochs: 1500,
        lr: 0.05,
        loss: Loss::Bce,
        optimizer: Optimizer::adam_default(),
        batch_size: 0,
        grad_clip: 0.0,
        lr_decay: 1.0,
    };
    let history = model.train(&x, &y, &cfg);

    println!("optimizador : Adam | loss: BCE");
    println!("loss inicial : {:.5}", history.first().unwrap());
    println!("loss final   : {:.5}", history.last().unwrap());
    println!("\npredicciones (esperado: 0, 1, 1, 0):");

    let pred = model.predict(&x);
    for i in 0..4 {
        let p = pred[[i, 0]];
        println!(
            "  [{}, {}] -> {:.4}  ({})",
            x[[i, 0]] as i32,
            x[[i, 1]] as i32,
            p,
            if p > 0.5 { 1 } else { 0 }
        );
    }
}
