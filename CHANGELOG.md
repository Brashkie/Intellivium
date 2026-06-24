# Changelog

Todos los cambios notables se documentan aquí.
Formato basado en [Keep a Changelog](https://keepachangelog.com/),
versionado [SemVer](https://semver.org/).

## [Unreleased]
### Planeado
- Optimizador Adam, losses BCE/Cross-Entropy, mini-batches, save/load.

## [0.1.0] - 2026-06-23
### Added
- Motor de autograd reverse-mode sobre tape (matmul, add+broadcast, relu, sigmoid, tanh, mse).
- Capa `Dense` (init He), `Model` secuencial, SGD.
- Bindings N-API (`neuroforge-napi`) y API TypeScript (`tensor`, `dense`, `Model`).
- Tests: gradient check + convergencia XOR (Rust), tests de `Tensor` y `Model` (Vitest).
- Tooling: Biome (lint/format), Vitest + coverage v8, CI en GitHub Actions.
