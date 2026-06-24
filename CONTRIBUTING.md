# Contribuir a NeuroForge

Gracias por el interés. Esta guía cubre cómo trabajar en el repo.

## Requisitos

- [Rust](https://rustup.rs/) (rustup)
- Node.js 18+
- `@napi-rs/cli` (se instala con `npm install`)

## Setup

```bash
git clone https://github.com/Brashkie/NeuroForge.git
cd NeuroForge
npm install
```

## Flujo de desarrollo

```bash
# motor Rust (sin Node)
cargo run --release -p neuroforge-core --example xor
cargo test -p neuroforge-core

# stack completo
npm run build        # native (.node) + TypeScript
npm test             # vitest
npm run coverage     # vitest + coverage v8
npm run lint         # biome
npm run typecheck    # tsc --noEmit
```

## Reglas de arquitectura

- **`crates/neuroforge-core`** es el motor puro-Rust. **No** debe contener nada de
  N-API, Node ni runtime. Mantenerlo testeable de forma aislada es deliberado.
- **`crates/neuroforge-napi`** es el único puente con Node. Solo cruzan tensores
  planos y ops de alto nivel.
- **`src/`** es la cara TypeScript. Sin lógica numérica pesada.

## Estilo

- Rust: `cargo fmt` + `cargo clippy -D warnings`.
- TS/JS: `npm run lint:fix` (Biome).
- Antes de abrir PR, corre lint, typecheck y tests.

## Commits

Se recomienda [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`.
