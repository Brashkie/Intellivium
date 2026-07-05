# NeuroForge — Stack y build

## Decisión de lenguajes (lo que preguntaste)

**Núcleo en Rust + N-API. Punto.** Es lo que ya haces en Kryx y es lo único
que distribuye bien por npm: binarios precompilados por plataforma, `npm install`
y listo, sin que el usuario instale nada raro.

### Por qué NO Julia ni C++ dentro del paquete npm

- **Julia**: se puede *embeber* (libjulia tiene API C), pero arrastra el runtime
  completo (VM + LLVM + stdlib = cientos de MB) y JIT en el primer uso. Empaquetar
  eso por plataforma en npm es inviable. Julia sirve como entorno de *research /
  prototipado* aparte, no como backend embebido de una lib de Node.
- **C/C++**: lo descartaste tú y está bien. Rust te da el mismo bajo nivel + SIMD
  sin el infierno de toolchains de C++. Todo el ecosistema ML puro-Rust
  (`ndarray`, y si luego quieres, `candle`/`burn`) cubre lo que harías con Eigen.

### Dónde entra Zig (opcional, después)

Zig SÍ encaja, pero **solo para kernels calientes** (matmul, conv, im2col) que Rust
enlace como `staticlib`. Compila a objeto nativo, sin runtime, y su SIMD es comodísimo.
Plan: cuando el perfil muestre que el matmul de `ndarray` es el cuello, escribes ese
kernel en Zig en `crates/neuroforge-napi` con un `build.rs` que invoque `zig build-lib`
y lo linkee. No lo metas al inicio: primero corre, luego optimiza.

## Capas del proyecto

```
crates/neuroforge-core   # motor puro-Rust: autograd (tape) + nn + optim  ← YA FUNCIONA
crates/neuroforge-napi   # bindings N-API -> .node
ts/                      # API pública (tensor, dense, Model)
examples/                # xor.rs (Rust) y xor.mjs (Node)
```

El grafo de autograd vive **entero en Rust**. Hacia JS solo cruzan tensores planos
(Float64Array + shape) y ops de alto nivel (construir, train, predict). Nada de
marshalear el grafo por operación.

## Estado actual (v0.1)

- ✅ Autograd reverse-mode (tape, sin Rc<RefCell>): matmul, add+broadcast, relu,
  sigmoid, tanh, MSE.
- ✅ Capa densa con init He, modelo secuencial, optimizadores SGD y Adam, losses MSE y BCE.
- ✅ XOR converge (loss 0.247 → 0.0002). Probado con `cargo run`.
- ⏳ Falta por hacer: mini-batches, save/load,
  Conv/RNN (el roadmap grande del README).

## Build

Requisitos: Rust (rustup), Node 18+, y `@napi-rs/cli` (ya en devDependencies).

```bash
# 1. probar SOLO el motor Rust (no necesita Node)
cargo run --release -p neuroforge-core --example xor

# 2. compilar todo (nativo + TS)
npm install
npm run build        # build:native (napi) + build:ts (tsc)

# 3. correr el ejemplo desde Node
npm test             # node examples/xor.mjs
```

`build:native` genera `index.js`, `index.d.ts` y el `.node` en la raíz.
`build:ts` empaqueta `src/` a `lib/` en ESM+CJS con **tsup** (con tipos .d.ts/.d.cts).

## Siguiente paso sugerido

Mini-batches + `save`/`load` de pesos. Con Adam + BCE ya listos (v0.2), el motor
entrena clasificación de verdad; falta escalarlo a datasets reales por lotes.
