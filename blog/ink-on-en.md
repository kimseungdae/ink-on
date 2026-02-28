---
title: "ink-on: Free, Private Handwriting Math Recognition in the Browser (No Server Required)"
description: "A zero-cost, open-source alternative to MyScript for EdTech — runs entirely client-side with 7.2 MB ONNX models"
tags: [webdev, machine-learning, edtech, privacy]
date: 2026-02-28
---

# ink-on: Free, Private Handwriting Math Recognition in the Browser

## The Problem: Privacy vs. Functionality in EdTech

If you're building an education app that recognizes handwritten math, your options are limited and expensive:

- **MyScript** — The industry standard. $500+/year licensing, all data sent to their servers.
- **Mathpix** — $0.01/request API. A classroom of 30 students writing 20 equations each = $6/day, $120/month.
- **Google Cloud Vision** — API costs + latency + your students' handwriting on Google's servers.

For education apps serving minors, this creates a compliance nightmare:

- **FERPA** (US) — Student education records cannot be disclosed to third parties without consent.
- **GDPR** (EU) — Processing children's personal data requires explicit parental consent.
- **PIPA** (South Korea) — Collection of personal information requires clear notice and consent.

Every API call that sends a student's handwriting to a third-party server is a potential compliance violation.

## The Solution: Privacy by Architecture

**ink-on** is a handwritten math recognition engine that runs **100% in the browser**. No server. No API key. No data ever leaves the device.

- **7.2 MB total** — INT8 quantized models, cached in IndexedDB after first download
- **Apache 2.0** — Free for commercial use, forever
- **Framework-agnostic** — Works with React, Vue, Svelte, vanilla JS
- **Offline capable** — Once models are cached, no internet needed

**[Try the live demo →](https://ink-on.vercel.app)**

Because recognition happens entirely on-device, there is no data to protect, no consent to obtain, no third-party to audit. Privacy compliance becomes architectural, not procedural.

## How It Works: CoMER in the Browser

ink-on runs [CoMER](https://github.com/Green-Wood/CoMER) (Coverage-guided Multi-scale Encoder-decoder Transformer), published at ECCV 2022, using ONNX Runtime Web.

### Architecture

```
Stroke[] → Preprocessing → Encoder (DenseNet + Transformer)
                                ↓
                           Decoder (Autoregressive Transformer)
                                ↓
                           Token IDs → LaTeX string
```

1. **Preprocessing** — User strokes are rendered on an offscreen canvas following [CROHME](https://www.isical.ac.in/~crohme/) conventions (white on black, top-left aligned). The image is scaled to 256px height with dynamic width aligned to 64px multiples.

2. **Encoder** — A DenseNet backbone extracts multi-scale features, followed by a Transformer encoder with positional embeddings.

3. **Decoder** — An autoregressive Transformer with coverage attention generates LaTeX tokens one at a time, using beam search with length normalization and repeat detection.

### INT8 Quantization: 92.8% Size Reduction

The original CoMER model is ~100 MB in FP32. We quantized it to INT8:

|                   | FP32 (original) | INT8 (ink-on) |
| ----------------- | --------------- | ------------- |
| Encoder           | ~55 MB          | 3.4 MB        |
| Decoder           | ~45 MB          | 4.0 MB        |
| **Total**         | **~100 MB**     | **7.2 MB**    |
| Browser feasible? | No              | Yes           |

This 92.8% size reduction makes the model practical for browser delivery while maintaining usable accuracy.

### Browser Optimizations

- **Dynamic input width** — Tensor width adapts to content. A simple "2" creates a 256×128 tensor instead of 256×1024, reducing computation by up to 80%.
- **Multi-threaded WASM** — SharedArrayBuffer enables parallel execution across CPU cores (2-4x speedup).
- **IndexedDB caching** — Models are downloaded once and cached locally. Subsequent visits load from cache in milliseconds.
- **Non-blocking decoding** — The decoder yields to the main thread every few steps, keeping the UI responsive.

## Benchmark: CROHME 2014

We evaluated ink-on on the [CROHME 2014](https://www.isical.ac.in/~crohme/) test set (986 handwritten math expressions).

| Model                     | ExpRate    | ≤1 edit    | ≤2 edits   | Size       | Runtime       |
| ------------------------- | ---------- | ---------- | ---------- | ---------- | ------------- |
| CoMER paper (FP32)        | 59.33%     | —          | —          | ~100 MB    | PyTorch/GPU   |
| **ink-on (INT8, beam=3)** | **36.41%** | **53.25%** | **65.82%** | **7.2 MB** | **ONNX/WASM** |

> **Note on INT8 accuracy**: The ~23% ExpRate drop compared to the original FP32 model reflects the combined effect of INT8 quantization (92.8% size reduction) and differences in the evaluation pipeline. ink-on runs end-to-end from InkML stroke coordinates through its own preprocessing, while the CoMER paper evaluates on pre-rendered images. The ≤2 edit accuracy of 65.82% shows the model produces nearly-correct output for most expressions.
>
> 125 of 986 samples (12.7%) contain symbols outside ink-on's 113-token vocabulary, making exact match impossible for those cases.

## Integration: 5 Minutes to Add

```bash
npm install ink-on onnxruntime-web
```

### Framework-Agnostic (React, Svelte, Vanilla JS)

```typescript
import { InferenceEngine, preprocessStrokes, loadVocab } from "ink-on/core";

const vocab = await loadVocab("/models/comer/vocab.json");
const engine = new InferenceEngine({
  encoderUrl: "/models/comer/encoder_int8.onnx",
  decoderUrl: "/models/comer/decoder_int8.onnx",
  beamWidth: 3,
});
await engine.init();

// Your canvas captures strokes → recognize
const input = preprocessStrokes(strokes);
const { latex } = await engine.recognize(input, vocab);
console.log(latex); // "x ^ { 2 } + 1"
```

### Vue 3

```vue
<script setup>
import {
  MathCanvas,
  InferenceEngine,
  preprocessStrokes,
  loadVocab,
} from "ink-on";
// ... (see README for full example)
</script>

<template>
  <MathCanvas @strokes-change="recognize" />
</template>
```

Download models from [GitHub Releases](https://github.com/kimseungdae/ink-on/releases) and place in `public/models/comer/`.

### Required Server Headers

ONNX Runtime Web's multi-threaded WASM requires these HTTP headers for SharedArrayBuffer:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Without them, inference still works but falls back to single-threaded (slower).

## Privacy Analysis for EdTech

### Why Client-Side Matters

| Regulation | Requirement                                       | ink-on Compliance           |
| ---------- | ------------------------------------------------- | --------------------------- |
| FERPA (US) | No disclosure of student records to third parties | ✓ No data transmission      |
| GDPR (EU)  | Data minimization, explicit consent for minors    | ✓ No data collection        |
| PIPA (KR)  | Notice and consent for personal data collection   | ✓ No collection occurs      |
| COPPA (US) | Parental consent for children under 13            | ✓ No data sent to operators |

With server-based recognition, every API call is a potential audit point. With ink-on, there is nothing to audit — the handwriting data never exists anywhere except the user's browser.

This is **privacy by architecture**: compliance isn't a configuration, a policy, or a checkbox — it's a structural property of the system.

### Cost Comparison

| Solution | Monthly cost (1,000 students, 50 expressions/day) | Data leaves device? |
| -------- | ------------------------------------------------- | ------------------- |
| MyScript | $500+ (license)                                   | Yes                 |
| Mathpix  | $15,000 ($0.01/req × 50K/day × 30)                | Yes                 |
| ink-on   | $0                                                | **No**              |

## What's Next

- **WebGPU support** — ONNX Runtime Web already supports WebGPU execution, offering 2-5x speedup on compatible browsers (Chrome 113+)
- **Higher accuracy models** — Exploring larger CoMER variants and training data augmentation
- **More math domains** — Extending vocabulary for chemistry, physics notation

## Links

- **Live Demo**: [ink-on.vercel.app](https://ink-on.vercel.app)
- **GitHub**: [github.com/kimseungdae/ink-on](https://github.com/kimseungdae/ink-on)
- **npm**: `npm install ink-on`
- **License**: Apache 2.0
- **Paper**: [CoMER (ECCV 2022)](https://github.com/Green-Wood/CoMER)

---

_ink-on is open source and contributions are welcome. If you're building EdTech tools and need handwriting recognition without the privacy headache, give it a try._
