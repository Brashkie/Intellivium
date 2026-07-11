# Changelog

Todos los cambios notables se documentan aquí.
Formato basado en [Keep a Changelog](https://keepachangelog.com/),
versionado [SemVer](https://semver.org/).

## [Unreleased]

## [0.4.0] - 2026-07-08
### Added
- **Softmax** (activación por filas, estable) y **Categorical Cross-Entropy (CCE)**:
  habilita clasificación **multiclase** (`loss: "cce"`, capa final `"softmax"`).
- **Gradient clipping** por norma L2 global (`gradClip`).
- **Learning-rate scheduler**: decaimiento exponencial por época (`lrDecay`).
- Tests (Rust y TS) de clasificación de 3 clases con softmax + CCE.

## [0.3.0] - 2026-07-05
### Added
- **Entrenamiento por mini-batches**: opción `batchSize` (barajado Fisher-Yates
  por época); `0`/ausente = batch completo.
- **`save` / `load` de modelos**: `model.save()` devuelve un estado JSON-friendly
  (arquitectura + pesos) y `Model.load(state)` lo reconstruye. En Rust:
  `Model::set_weights` y `Activation::as_str`.
- Tests nuevos (Rust y TS) para mini-batches y round-trip de save/load.

## [0.2.1] - 2026-07-05
### Fixed
- Release: se eliminó la doble publicación de sub-paquetes en `release.yml`
  (el loop manual + `napi prepublish` chocaban → error 403 "already published").
  Ahora `napi prepublish` (vía `prepublishOnly`) publica los sub-paquetes y fija
  las `optionalDependencies`, y `npm publish` sube el paquete principal.
### Changed
- **Renombrado de NeuroForge a Intellivium.** El paquete npm es `intellivium`
  (sin scope). El repo también pasó de `NeuroForge` a `Intellivium`.
- **Relicenciado a Apache-2.0** (antes propietario).
- Publicación multiplataforma vía GitHub Actions (`release.yml`): binarios
  prebuilt por plataforma como sub-paquetes `intellivium-<triple>`.
- El binding nativo se genera en la raíz como `index.cjs` + `.node`.
- Build del SDK con **tsup** (bundle dual ESM + CJS + tipos).
- Binding nativo emitido como `index.cjs` (evita choque con `"type": "module"`).
- Carga perezosa del addon + más tests unitarios (coverage de `index`/`layers`/`tensor` al 100%).

### Planeado
- Mini-batches y data loaders, save/load de modelos, capas Conv/RNN.

## [0.2.0] - 2026-06-24
### Added
- Optimizador **Adam** (con estado de momentos por capa).
- Loss **BCE** (binary cross-entropy) con clamp por estabilidad numérica.
- API de entrenamiento por configuración: `optimizer` ("sgd"|"adam") y
  `loss` ("mse"|"bce"), más hiperparámetros de Adam (`beta1`, `beta2`, `eps`).
- Test de convergencia XOR con Adam + BCE (Rust).
### Changed
- `Model.train` ahora recibe `TrainConfig` (Rust) / `TrainOptions` (TS) en vez de
  `(epochs, lr)`. **Breaking** respecto a 0.1.0 (aún sin publicar en npm).

## [0.1.0] - 2026-06-23
### Added
- Motor de autograd reverse-mode sobre tape (matmul, add+broadcast, relu, sigmoid, tanh, mse).
- Capa `Dense` (init He), `Model` secuencial, SGD.
- Bindings N-API (`neuroforge-napi`) y API TypeScript (`tensor`, `dense`, `Model`).
- Tests: gradient check + convergencia XOR (Rust), tests de `Tensor` y `Model` (Vitest).
- Tooling: Biome (lint/format), Vitest + coverage v8, CI en GitHub Actions.
