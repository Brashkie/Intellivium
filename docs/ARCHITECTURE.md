# Arquitectura

```
src/ (TypeScript)
  index.ts      barrel de exports
  tensor.ts     Tensor + tensor()
  layers.ts     dense() + tipos
  model.ts      Model (train/predict)
  native.ts     shim del addon nativo
        │  Float64Array + shape
        ▼
crates/neuroforge-napi   puente N-API (cdylib → .node)
        │
        ▼
crates/neuroforge-core   motor puro-Rust
  tape.rs   autograd reverse-mode (Wengert list)
  nn.rs     Dense, Model, SGD
  rng.rs    RNG xorshift + Box-Muller
```

## Decisiones clave

1. **El grafo de autograd vive 100% en Rust.** Hacia JS solo cruzan tensores
   planos. Nunca se marshalea el grafo por operación (sería lento y frágil).
2. **Un solo lenguaje nativo: Rust.** Sin Julia (runtime gigante + JIT, no
   empaquetable en npm) ni C++ (innecesario; Rust cubre bajo nivel + SIMD).
3. **Zig reservado para kernels calientes**, enlazados como `staticlib` cuando
   el perfilado lo justifique. No al inicio.
4. **f32 en el core, f64 en la frontera JS** (los números de JS son f64); la
   conversión ocurre en el puente N-API.
