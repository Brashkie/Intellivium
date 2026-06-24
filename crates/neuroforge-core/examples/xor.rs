//! Prueba real del motor: aprender XOR (problema no lineal clásico).
//! Si esto converge, el autograd + backprop están bien.

use ndarray::array;
use neuroforge_core::{Activation, Dense, Model, Rng};

fn main() {
    let mut rng = Rng::new(42);

    let x = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let y = array![[0.0], [1.0], [1.0], [0.0]];

    let mut model = Model::new(vec![
        Dense::new(2, 8, Activation::Tanh, &mut rng),
        Dense::new(8, 1, Activation::Sigmoid, &mut rng),
    ]);

    let history = model.train(&x, &y, 4000, 0.5);

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
