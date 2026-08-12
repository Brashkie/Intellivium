# Changelog

Todos los cambios notables se documentan aquí.
Formato basado en [Keep a Changelog](https://keepachangelog.com/),
versionado [SemVer](https://semver.org/).

## [Unreleased]
### Tests
- Cobertura completa de `importWeights`: casos de nº de capas distinto, tipo de
  capa distinto, y transferencia de una capa BatchNorm (cierra el hueco de
  cobertura en `model.ts`).


## [0.9.0] - 2026-08-11
### Added
- **Custom Dataset**: interfaz `Dataset` (`length` + `get(i)`) que cualquier
  fuente puede implementar (en memoria, generada, perezosa…). `DataLoader` ahora
  acepta cualquier `Dataset`, no solo `TensorDataset`.
- **`exportWeights()` / `importWeights()`**: variante solo-pesos de save/load.
  Transfiere parámetros entre modelos de la misma arquitectura (fine-tuning,
  weight sharing) con validación de nº y tipo de capas.
- Tests de dataset generado y de transferencia de pesos.
### Notes
- Con esto la **Fase 2 (Training) queda completa** ✅.

## [0.8.0] - 2026-08-07
### Added
- **BatchNorm** (`batchNorm(features)`): normalización por columna sobre el batch,
  con **estadísticas móviles** (running mean/var, momentum 0.1) al estilo PyTorch.
  En training usa stats del batch y actualiza las running; en `predict`/`evaluate`
  usa las running (inferencia estable e independiente del batch).
- `save`/`load` serializan también las running stats de BatchNorm, así que un
  modelo cargado infiere idéntico.
- Tests (Rust y TS) de BatchNorm: entrena, actualiza running stats y round-trip.

## [0.7.1] - 2026-08-03
### Added
- **Mejoras de Fase 1 — más operaciones de `Tensor`** (solo JS, sin tocar el motor):
  - Factories: `Tensor.zeros`, `ones`, `full`, `eye`.
  - Aritmética elemento a elemento con broadcasting de vector fila/columna y
    escalares: `add`, `sub`, `mul`, `scale`, `addScalar`, `neg`, `map`.
  - `matmul` (utilidad en JS, con verificación de shapes).
  - Reducciones: `sum`, `mean`, `max`, `min`, `argmaxRows`.
  - Utilidades: `clone`, `equals` (con tolerancia).
- Tests de todo lo anterior (`tensor.ts` al 100% de cobertura).

## [0.7.0] - 2026-07-25
### Added
- **Fase 3 — comienzo de la biblioteca neural.** El modelo pasa de una lista de
  capas densas a un sistema de capas (`Layer`), habilitando capas sin pesos y de
  normalización dentro del `Sequential`.
- **Dropout** (`dropout(p)`): inverted dropout, activo solo en entrenamiento;
  en `predict`/`evaluate` es identidad.
- **LayerNorm** (`layerNorm(features)`): normalización por muestra con gamma/beta
  entrenables (backward vectorizado, media~0 y var~1 por fila).
- `save`/`load` ahora serializan el tipo de capa (`kind`) y sus parámetros.
- Tests (Rust y TS) de Dropout y LayerNorm.
### Changed
- API interno del núcleo: `Model` contiene `Vec<Layer>` en vez de `Vec<Dense>`.
  El API público de TS es compatible (dense/dropout/layerNorm).

## [0.6.0] - 2026-07-23
### Added
- **Nuevas activaciones**: `leakyrelu`, `elu`, `gelu` (aprox. sigmoide).
- **Nuevas losses**: `mae` (L1) y `huber` (smooth L1).
- **Tensor views**: `reshape` (comparte buffer, admite `-1`), `transpose`,
  `slice` y `row` (vistas), y `at(i, j)`.
- Tests (Rust y TS) para las nuevas activaciones, losses y vistas.
### Notes
- Con esto la **Fase 1** queda completa salvo el sistema de múltiples dtypes
  (el motor es f32 a propósito, el estándar en ML).

## [0.5.0] - 2026-07-11
### Added
- **Validación durante el entrenamiento**: `model.fit(x, y, opts, { x, y })`
  devuelve `history`, `valHistory`, `bestEpoch`, `bestLoss` y `stoppedEarly`.
- **Early stopping** por paciencia (`patience`, `minDelta`).
- **Checkpoints**: `restoreBest` restaura los pesos de la mejor época al terminar.
- **`model.evaluate(x, y, loss)`**: calcula la loss sin entrenar.
- **`TensorDataset`** (con `select` y `split` determinista) y **`DataLoader`**
  (iterable por lotes, con shuffle opcional).
- En Rust: `Model::train_with_validation`, `Model::evaluate` y `TrainResult`.

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
