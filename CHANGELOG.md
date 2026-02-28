# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-02-28

Initial release of ink-on.

### Added

- Browser-based handwritten math recognition using CoMER (ECCV 2022) ONNX models
- Framework-agnostic `InferenceEngine` core — usable from React, Svelte, vanilla JS, or any framework
- Vue 3 `<MathCanvas>` component with mouse + touch support and smooth Bezier strokes
- INT8 quantized models: encoder (3.4 MB) + decoder (4.0 MB), 92.8% size reduction from FP32
- Beam search decoding with adaptive width and length normalization
- Greedy decoding mode for low-end devices
- Repeat detection preventing degenerate decoder output
- Meaningful stroke filtering (minimum bounding box, point count, path length)
- IndexedDB model caching — instant reload after first 7.2 MB download
- Offline recognition once models are cached
- KaTeX LaTeX rendering with copy-to-clipboard
- npm package with dual exports: `ink-on` (Vue) and `ink-on/core` (framework-agnostic)
- Apache 2.0 license
- Bilingual README (English + Korean)
- Live demo at https://ink-on.vercel.app

### Performance

- WASM multi-threading via SharedArrayBuffer for parallel execution across CPU cores
- Dynamic input tensor width (64px-aligned) — up to 80% computation reduction for simple expressions
- Non-blocking decoding with main thread yielding every few steps
- ONNX Runtime Web externalized from bundle to prevent Worker context errors

[0.1.0]: https://github.com/kimseungdae/ink-on/releases/tag/v0.1.0
