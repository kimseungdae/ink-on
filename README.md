# ink-on

> Handwritten math expression recognition running entirely in the browser.
> Framework-agnostic ONNX inference engine + Vue 3 canvas component, powered by [CoMER](https://github.com/Green-Wood/CoMER) (ECCV 2022).

[![npm version](https://img.shields.io/npm/v/ink-on.svg)](https://www.npmjs.com/package/ink-on)
[![license](https://img.shields.io/npm/l/ink-on.svg)](./LICENSE)

**[Live Demo](https://ink-on.vercel.app)** | [English](#features) | [한국어](#한국어)

---

## Features

- **100% client-side** — Runs entirely in the browser via ONNX Runtime Web (WASM/WebGPU). No server, no API key.
- **Tiny models** — INT8 quantized encoder (3.4 MB) + decoder (4.0 MB), total **7.2 MB**.
- **Web Worker inference** — ONNX inference runs off the main thread, keeping the UI responsive during recognition.
- **LaTeX auto-repair** — Automatic brace balancing and `\frac`/`\sqrt` argument fixing with KaTeX runtime validation.
- **Recognition modes** — Auto, Number (digits + basic operators), and Expression mode with vocabulary masking.
- **Framework-agnostic core** — Use `InferenceEngine` standalone from React, Svelte, vanilla JS, or any framework.
- **Vue 3 component** — Drop-in `<MathCanvas>` with mouse + touch support, smooth Bézier strokes, and responsive sizing.
- **Beam search decoding** — Adaptive beam width based on device capability for quality/speed balance.
- **IndexedDB caching** — Models are cached locally after first download; instant reload on revisit.
- **PWA ready** — Installable as a standalone app with Web App Manifest.
- **Tested** — 51 unit tests across 4 test suites with Vitest. CI/CD via GitHub Actions.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  User draws on <MathCanvas> (or your own canvas)             │
│  Stroke[] = [{ points: [{x, y}, ...], lineWidth: 3 }, ...]  │
└───────────────────┬──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│  preprocessStrokes(strokes)                     [Main Thread]│
│  ┌────────────────────────────────────────────────────────┐  │
│  │ 1. Resample stroke points at uniform 3px intervals     │  │
│  │ 2. Render with Bézier curves (white on black canvas)   │  │
│  │ 3. Scale to height=256, dynamic width (64px-aligned)   │  │
│  │ 4. Convert to grayscale Float32 tensor + padding mask   │  │
│  └────────────────────────────────────────────────────────┘  │
│  → PreprocessResult { tensor, mask, height, width }          │
└───────────────────┬──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│  Web Worker (inference.worker.ts)                            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ ENCODER (CoMER — DenseNet + Transformer)               │  │
│  │  Input:  pixel_values [1,1,H,W] + pixel_mask [1,H,W]  │  │
│  │  Output: encoder_features + encoder_mask               │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │ DECODER (Autoregressive Transformer)                   │  │
│  │  Greedy (beam=1) or Beam search (beam=2-3)             │  │
│  │  Mode-based vocabulary masking (auto/number/expression) │  │
│  │  Repeat detection + length normalization                │  │
│  └────────────────────────────────────────────────────────┘  │
│  → candidates: [{ ids, logProb }, ...]                       │
└───────────────────┬──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│  Post-processing + Validation                   [Main Thread]│
│  ┌────────────────────────────────────────────────────────┐  │
│  │ 1. decodeToTokenArray(ids, vocab)                      │  │
│  │ 2. repairLatex(tokens) — brace balancing, arg fixing   │  │
│  │ 3. isCompleteExpression(tokens) — completeness check   │  │
│  │ 4. KaTeX validation — katex.renderToString(throwOnError)│  │
│  │ 5. Select first valid candidate from beam results       │  │
│  └────────────────────────────────────────────────────────┘  │
│  → RecognitionResult { latex, tokenIds, encoderMs, ... }     │
└───────────────────┬──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│  Render LaTeX with KaTeX, MathJax, or your choice            │
└──────────────────────────────────────────────────────────────┘
```

### Key design insights

- **Off-main-thread inference** — ONNX encoder + decoder run in a Web Worker. The main thread stays responsive for drawing and UI updates during the 1-2 second inference.
- **Beam search → validation pipeline** — The Worker returns top-N beam candidates. The main thread runs each through `repairLatex()` → `isCompleteExpression()` → KaTeX validation, selecting the first valid result.
- **Dynamic input width** — The encoder tensor width adapts to the drawn content (aligned to 64px multiples). A simple "2" creates a 256×128 tensor instead of 256×1024, reducing encoder computation by up to 80%.
- **Stroke resampling + Bézier** — Raw touch points are resampled at uniform 3px intervals, then rendered with quadratic Bézier curves for smooth, consistent strokes matching CROHME training data.
- **CROHME convention** — Preprocessing follows the CROHME handwriting dataset format (white strokes on black background, top-left alignment) that the CoMER model was trained on.
- **LaTeX auto-repair** — `repairLatex()` fixes common decoder errors: unbalanced braces, extra `\frac`/`\sqrt` arguments. Deterministic, no model dependency.
- **Vocabulary masking** — Number mode restricts decoder output to digits, basic operators, and structural tokens, eliminating impossible symbols from beam search.
- **Repeat detection** — Both greedy and beam decoders detect and stop on repeated tokens, preventing garbage output like "EEEEE..." from ambiguous input.
- **IndexedDB model cache** — `fetchWithCache()` stores downloaded ONNX models in IndexedDB. Subsequent page loads skip the 7.2 MB download entirely.
- **Multi-threaded WASM** — Uses `SharedArrayBuffer` with COOP/COEP headers for parallel WASM execution across multiple CPU cores.

---

## Quick Start

### Installation

```bash
npm install ink-on
```

You also need the peer dependencies:

```bash
npm install vue onnxruntime-web
```

### Download Models

The ONNX models are **not included** in the npm package. Download and place them in your app's `public/` directory:

| File                | Size   | Description                                            |
| ------------------- | ------ | ------------------------------------------------------ |
| `encoder_int8.onnx` | 3.4 MB | CoMER encoder (DenseNet + Transformer), INT8 quantized |
| `decoder_int8.onnx` | 4.0 MB | CoMER autoregressive decoder, INT8 quantized           |
| `vocab.json`        | 4 KB   | Token vocabulary (245 symbols)                         |

Place them at `public/models/comer/` or any path you choose. You can download them from the [GitHub repository releases](https://github.com/kimseungdae/ink-on/releases).

### Vue 3 Usage

```vue
<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";
import {
  MathCanvas,
  InferenceEngine,
  preprocessStrokes,
  isStrokeMeaningful,
  loadVocab,
} from "ink-on";
import type { Stroke, RecognitionResult, Vocab } from "ink-on";

const canvasRef = ref<InstanceType<typeof MathCanvas> | null>(null);
const result = ref<RecognitionResult | null>(null);

let engine: InferenceEngine;
let vocab: Vocab;

onMounted(async () => {
  vocab = await loadVocab("/models/comer/vocab.json");
  engine = new InferenceEngine({
    encoderUrl: "/models/comer/encoder_int8.onnx",
    decoderUrl: "/models/comer/decoder_int8.onnx",
    beamWidth: 3,
  });
  await engine.init();
});

onUnmounted(() => {
  engine?.dispose();
});

async function onStrokesChange(strokes: Stroke[]) {
  if (!isStrokeMeaningful(strokes)) return;
  const input = preprocessStrokes(strokes);
  result.value = await engine.recognize(input, vocab);
}
</script>

<template>
  <MathCanvas
    ref="canvasRef"
    :width="700"
    :height="300"
    :line-width="3"
    @strokes-change="onStrokesChange"
  />
  <pre v-if="result">{{ result.latex }}</pre>
</template>
```

### Framework-Agnostic Usage (React, Svelte, Vanilla JS)

Import from `ink-on/core` to avoid the Vue dependency:

```typescript
import {
  InferenceEngine,
  preprocessStrokes,
  isStrokeMeaningful,
  loadVocab,
} from "ink-on/core";
import type { Stroke } from "ink-on/core";

// Initialize once
const vocab = await loadVocab("/models/comer/vocab.json");
const engine = new InferenceEngine({
  encoderUrl: "/models/comer/encoder_int8.onnx",
  decoderUrl: "/models/comer/decoder_int8.onnx",
  beamWidth: 3,
  executionProvider: "wasm",
});
await engine.init();

// Recognize from your own canvas/stroke capture
const strokes: Stroke[] = [
  {
    points: [
      { x: 10, y: 20 },
      { x: 30, y: 50 } /* ... */,
    ],
    lineWidth: 3,
  },
];

if (isStrokeMeaningful(strokes)) {
  const input = preprocessStrokes(strokes);
  const result = await engine.recognize(input, vocab);
  console.log(result.latex); // "x ^ { 2 } + y ^ { 2 }"
  console.log(result.totalMs); // 450
}

// Cleanup when done
engine.dispose();
```

---

## Model Hosting

### Required Server Headers

ONNX Runtime Web uses multi-threaded WASM via `SharedArrayBuffer`, which requires these HTTP headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

**Vite example** (`vite.config.ts`):

```typescript
export default defineConfig({
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
```

Without these headers, inference will still work but falls back to single-threaded execution (slower).

### Hosting Options

1. **Static files** — Place models in your app's `public/models/comer/` directory
2. **CDN** — Upload to CloudFlare R2, AWS S3, or any CDN with CORS support
3. **GitHub Releases** — Attach as release assets

If models are served from a different origin, ensure the CDN sends appropriate `Access-Control-Allow-Origin` headers.

---

## API Reference

### `<MathCanvas>` Component

Vue 3 component for capturing handwritten strokes with mouse and touch support.

#### Props

| Prop          | Type     | Default     | Description                       |
| ------------- | -------- | ----------- | --------------------------------- |
| `width`       | `number` | `600`       | Canvas resolution width (pixels)  |
| `height`      | `number` | `300`       | Canvas resolution height (pixels) |
| `lineWidth`   | `number` | `3`         | Stroke line width                 |
| `strokeColor` | `string` | `'#000000'` | Stroke color                      |

#### Events

| Event            | Payload    | Description                                                    |
| ---------------- | ---------- | -------------------------------------------------------------- |
| `strokes-change` | `Stroke[]` | Emitted when a stroke ends (all strokes including the new one) |

#### Exposed Methods

| Method    | Description                          |
| --------- | ------------------------------------ |
| `clear()` | Remove all strokes and clear canvas  |
| `undo()`  | Remove the last stroke               |
| `strokes` | Reactive ref to current stroke array |

### `InferenceEngine`

Main ONNX inference engine. Framework-agnostic.

#### Constructor

```typescript
new InferenceEngine(options: InferenceEngineOptions)
```

| Option              | Type                 | Default    | Description                            |
| ------------------- | -------------------- | ---------- | -------------------------------------- |
| `encoderUrl`        | `string`             | _required_ | URL to the encoder ONNX model          |
| `decoderUrl`        | `string`             | _required_ | URL to the decoder ONNX model          |
| `maxDecodeSteps`    | `number`             | `50`       | Maximum autoregressive decode steps    |
| `beamWidth`         | `number`             | `3`        | Beam search width (1 = greedy, faster) |
| `executionProvider` | `'wasm' \| 'webgpu'` | `'wasm'`   | ONNX execution provider                |

#### Methods

| Method                          | Returns                      | Description                                      |
| ------------------------------- | ---------------------------- | ------------------------------------------------ |
| `init()`                        | `Promise<void>`              | Load and initialize ONNX sessions (lazy, cached) |
| `recognize(input, vocab, mode)` | `Promise<RecognitionResult>` | Run encoder + decoder inference with mode        |
| `dispose()`                     | `void`                       | Release ONNX sessions and free memory            |

`mode` is an optional `RecognitionMode` parameter: `'auto'` (default), `'number'`, or `'expression'`.

### `RecognitionResult`

```typescript
interface RecognitionResult {
  latex: string; // Decoded LaTeX string (e.g., "x ^ { 2 } + 1")
  tokenIds: number[]; // Raw token IDs from decoder
  encoderMs: number; // Encoder inference time (ms)
  decoderMs: number; // Decoder inference time (ms)
  totalMs: number; // Total inference time (ms)
}
```

### `Stroke` / `StrokePoint`

```typescript
interface StrokePoint {
  x: number;
  y: number;
}

interface Stroke {
  points: StrokePoint[];
  lineWidth: number;
}
```

### Preprocessing

```typescript
// Convert strokes to a normalized tensor for the encoder
preprocessStrokes(strokes: Stroke[]): PreprocessResult

// Check if strokes have enough content for meaningful recognition
// Filters out dots, accidental taps, and trivial input
isStrokeMeaningful(strokes: Stroke[]): boolean
```

### Tokenizer

```typescript
// Load vocabulary from a JSON file (cached after first call)
loadVocab(url: string): Promise<Vocab>

// Convert token IDs to a LaTeX string
decodeTokenIds(ids: number[], vocab: Vocab): string

// Convert token IDs to an array of LaTeX tokens (filters special tokens)
decodeToTokenArray(ids: number[], vocab: Vocab): string[]
```

### LaTeX Repair

```typescript
// Auto-repair common decoder errors: unbalanced braces, extra \frac/\sqrt args
repairLatex(tokens: string[]): string[]

// Check if a token array forms a complete math expression
// Returns false for lone \frac, \sqrt, \sum without required arguments
isCompleteExpression(tokens: string[]): boolean
```

### Model Cache (IndexedDB)

```typescript
// Fetch a model with IndexedDB caching (recommended)
fetchWithCache(url: string): Promise<ArrayBuffer>

// Direct IndexedDB access
getCachedModel(url: string): Promise<ArrayBuffer | null>
cacheModel(url: string, data: ArrayBuffer): Promise<void>
```

---

## Configuration Guide

### Beam Width

| Beam Width  | Speed    | Quality | Recommended For                    |
| ----------- | -------- | ------- | ---------------------------------- |
| 1 (greedy)  | Fastest  | Good    | Low-end devices, real-time preview |
| 2           | Fast     | Better  | Mid-range devices                  |
| 3 (default) | Moderate | Best    | Desktop browsers                   |

### Execution Provider

| Provider | Support                | Performance | Notes                       |
| -------- | ---------------------- | ----------- | --------------------------- |
| `wasm`   | All modern browsers    | Good        | Default, universal fallback |
| `webgpu` | Chrome 113+, Edge 113+ | 2-5x faster | Auto-falls back to WASM     |

### Adaptive Configuration

```typescript
const cores = navigator.hardwareConcurrency || 2;
const mem = (navigator as any).deviceMemory || 4;

const engine = new InferenceEngine({
  encoderUrl: "/models/comer/encoder_int8.onnx",
  decoderUrl: "/models/comer/decoder_int8.onnx",
  beamWidth: mem <= 2 ? 1 : cores <= 4 ? 2 : 3,
  executionProvider: navigator.gpu ? "webgpu" : "wasm",
});
```

---

## Performance Tips

- **Web Worker** — The demo app runs inference in a Web Worker to avoid blocking the UI. Use `inference.worker.ts` as a reference.
- **Debounce recognition** — Don't call `recognize()` on every stroke. Wait 1-2 seconds after the user stops drawing.
- **Use `isStrokeMeaningful()`** — Skip inference for dots and accidental taps.
- **Number mode** — Use `'number'` mode for digit-only input. Vocabulary masking significantly reduces decoder search space.
- **Preload models** — Call `engine.init()` early (on mount) to overlap loading with user interaction.
- **Model caching** — `fetchWithCache()` stores models in IndexedDB. First visit downloads 7.2 MB; revisits load instantly from cache.
- **Call `dispose()`** — Release ONNX sessions when unmounting to free WASM memory.
- **COOP/COEP headers** — Without these, WASM runs single-threaded. Multi-threading can be 2-4x faster on multi-core CPUs.

---

## Browser Requirements

| Feature           | Minimum                               | Notes                     |
| ----------------- | ------------------------------------- | ------------------------- |
| WebAssembly       | All modern browsers                   | Core requirement          |
| WASM SIMD         | Chrome 91+, Firefox 89+, Safari 16.4+ | Enabled by default        |
| SharedArrayBuffer | Requires COOP/COEP headers            | For multi-threaded WASM   |
| IndexedDB         | All modern browsers                   | For model caching         |
| Canvas 2D         | All modern browsers                   | For stroke preprocessing  |
| WebGPU (optional) | Chrome 113+, Edge 113+                | Faster execution provider |

---

## How It Works

This library runs [CoMER](https://github.com/Green-Wood/CoMER) (Coverage-guided Multi-scale Encoder-decoder Transformer, ECCV 2022) entirely in the browser using ONNX Runtime Web.

### Model Pipeline

1. **Stroke capture** — Canvas captures pointer/touch events as `Stroke[]` with coordinates and line width.
2. **Preprocessing** — Strokes are rendered to an offscreen canvas following [CROHME](https://www.isical.ac.in/~crohme/) training conventions (white strokes on black background, top-left aligned). The image is scaled to 256px height with dynamic width aligned to 64px multiples, then converted to a grayscale float32 tensor with a binary padding mask.
3. **Encoder** — DenseNet backbone extracts multi-scale visual features. A Transformer encoder produces contextual feature maps with position embeddings.
4. **Decoder** — An autoregressive Transformer decoder with coverage attention generates token IDs one at a time. Beam search with length normalization selects the best hypothesis. Repeat detection halts degenerate sequences.
5. **Tokenizer** — Token IDs are mapped to LaTeX symbols via `vocab.json` (245 symbols including digits, operators, Greek letters, and structural tokens like fractions and superscripts).

### Optimizations

- **INT8 quantization** — Models are quantized from FP32 to INT8, reducing size from ~100 MB to 7.2 MB with minimal accuracy loss.
- **Web Worker inference** — ONNX inference runs off the main thread via Web Worker, preventing UI blocking during 1-2s recognition.
- **Dynamic input width** — Tensor width adapts to content, avoiding wasted computation on padding. Simple expressions run up to 80% faster.
- **Stroke resampling** — Raw touch points are resampled at uniform 3px intervals and rendered with Bézier curves, matching CROHME training data quality.
- **LaTeX auto-repair** — `repairLatex()` fixes unbalanced braces and excess `\frac`/`\sqrt` arguments. KaTeX runtime validation selects the best beam candidate.
- **Vocabulary masking** — Number mode restricts decoder logits to digits + basic operators, eliminating impossible tokens from beam search.
- **Multi-threaded WASM** — `SharedArrayBuffer` enables parallel execution across CPU cores.
- **IndexedDB caching** — Models are downloaded once and cached locally for instant reload.

---

## Benchmark

Evaluated on the [CROHME 2014](https://www.isical.ac.in/~crohme/) test set (986 handwritten math expressions), end-to-end from InkML strokes through ink-on's full pipeline.

| Model                     | ExpRate    | ≤1 edit    | ≤2 edits   | Size       |
| ------------------------- | ---------- | ---------- | ---------- | ---------- |
| CoMER paper (FP32)        | 59.33%     | —          | —          | ~100 MB    |
| **ink-on (INT8, beam=3)** | **36.41%** | **53.25%** | **65.82%** | **7.2 MB** |

The ExpRate gap reflects INT8 quantization (92.8% size reduction), preprocessing differences, and limited vocabulary (113 tokens). See [benchmark/](./benchmark/) for reproduction scripts.

---

## License

[Apache License 2.0](./LICENSE) — Copyright 2025 kimseungdae

---

# 한국어

> 브라우저에서 완전히 실행되는 손글씨 수학 수식 인식 라이브러리.
> 프레임워크 독립 ONNX 추론 엔진 + Vue 3 캔버스 컴포넌트. [CoMER](https://github.com/Green-Wood/CoMER) (ECCV 2022) 기반.

**[라이브 데모](https://ink-on.vercel.app)**

## 기능

- **100% 클라이언트 사이드** — ONNX Runtime Web(WASM/WebGPU)으로 브라우저에서 완전 실행. 서버 없음, API 키 없음.
- **경량 모델** — INT8 양자화 인코더(3.4MB) + 디코더(4.0MB), 총 **7.2MB**.
- **Web Worker 추론** — ONNX 추론이 별도 Worker 스레드에서 실행되어 인식 중 UI 블로킹 없음.
- **LaTeX 자동 수정** — 괄호 균형 맞춤, `\frac`/`\sqrt` 인수 자동 수정 + KaTeX 런타임 검증.
- **인식 모드** — Auto, Number(숫자+기본연산자), Expression 모드. 어휘 마스킹으로 검색 공간 축소.
- **프레임워크 독립 코어** — React, Svelte, 바닐라 JS 등 어떤 프레임워크에서든 `InferenceEngine` 단독 사용 가능.
- **Vue 3 컴포넌트** — 마우스 + 터치 지원 `<MathCanvas>` 드롭인 컴포넌트, 부드러운 Bézier 스트로크, 반응형 크기 조정.
- **빔 서치 디코딩** — 디바이스 성능에 따른 적응형 빔 폭으로 품질/속도 균형 조절.
- **IndexedDB 캐싱** — 첫 다운로드 후 모델을 로컬에 캐시, 재방문 시 즉시 로드.
- **PWA 지원** — Web App Manifest로 독립 앱 설치 가능.
- **테스트 완비** — Vitest 기반 51개 단위 테스트, GitHub Actions CI/CD.

## 빠른 시작

### 설치

```bash
npm install ink-on
```

피어 의존성도 필요합니다:

```bash
npm install vue onnxruntime-web
```

### 모델 다운로드

ONNX 모델은 npm 패키지에 **포함되지 않습니다**. 다운로드하여 앱의 `public/` 디렉토리에 배치하세요:

| 파일                | 크기   | 설명                                               |
| ------------------- | ------ | -------------------------------------------------- |
| `encoder_int8.onnx` | 3.4 MB | CoMER 인코더 (DenseNet + Transformer), INT8 양자화 |
| `decoder_int8.onnx` | 4.0 MB | CoMER 오토리그레시브 디코더, INT8 양자화           |
| `vocab.json`        | 4 KB   | 토큰 어휘 (245개 심볼)                             |

`public/models/comer/`에 배치하거나 원하는 경로에 놓으세요. [GitHub 리포지토리 릴리즈](https://github.com/kimseungdae/ink-on/releases)에서 다운로드할 수 있습니다.

### Vue 3 사용법

```vue
<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";
import {
  MathCanvas,
  InferenceEngine,
  preprocessStrokes,
  isStrokeMeaningful,
  loadVocab,
} from "ink-on";
import type { Stroke, RecognitionResult, Vocab } from "ink-on";

const canvasRef = ref<InstanceType<typeof MathCanvas> | null>(null);
const result = ref<RecognitionResult | null>(null);

let engine: InferenceEngine;
let vocab: Vocab;

onMounted(async () => {
  vocab = await loadVocab("/models/comer/vocab.json");
  engine = new InferenceEngine({
    encoderUrl: "/models/comer/encoder_int8.onnx",
    decoderUrl: "/models/comer/decoder_int8.onnx",
    beamWidth: 3,
  });
  await engine.init();
});

onUnmounted(() => {
  engine?.dispose();
});

async function onStrokesChange(strokes: Stroke[]) {
  if (!isStrokeMeaningful(strokes)) return;
  const input = preprocessStrokes(strokes);
  result.value = await engine.recognize(input, vocab);
}
</script>

<template>
  <MathCanvas
    ref="canvasRef"
    :width="700"
    :height="300"
    :line-width="3"
    @strokes-change="onStrokesChange"
  />
  <pre v-if="result">{{ result.latex }}</pre>
</template>
```

### 프레임워크 독립 사용법 (React, Svelte, Vanilla JS)

Vue 의존성 없이 `ink-on/core`에서 import:

```typescript
import {
  InferenceEngine,
  preprocessStrokes,
  isStrokeMeaningful,
  loadVocab,
} from "ink-on/core";
import type { Stroke } from "ink-on/core";

const vocab = await loadVocab("/models/comer/vocab.json");
const engine = new InferenceEngine({
  encoderUrl: "/models/comer/encoder_int8.onnx",
  decoderUrl: "/models/comer/decoder_int8.onnx",
  beamWidth: 3,
});
await engine.init();

// 커스텀 캔버스/스트로크 캡처에서 인식
const strokes: Stroke[] = [
  {
    points: [
      { x: 10, y: 20 },
      { x: 30, y: 50 },
    ],
    lineWidth: 3,
  },
];

if (isStrokeMeaningful(strokes)) {
  const input = preprocessStrokes(strokes);
  const result = await engine.recognize(input, vocab);
  console.log(result.latex); // "x ^ { 2 } + y ^ { 2 }"
  console.log(result.totalMs); // 450
}

engine.dispose();
```

## 모델 호스팅

### 필수 서버 헤더

ONNX Runtime Web은 `SharedArrayBuffer`를 통한 멀티스레드 WASM을 사용하며, 다음 HTTP 헤더가 필요합니다:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

이 헤더 없이도 동작하지만 싱글스레드로 폴백되어 느려집니다.

### 호스팅 옵션

1. **정적 파일** — 앱의 `public/models/comer/` 디렉토리에 배치
2. **CDN** — CloudFlare R2, AWS S3 등 CORS 지원 CDN에 업로드
3. **GitHub Releases** — 릴리즈 에셋으로 첨부

## API 레퍼런스

영문 [API Reference](#api-reference) 섹션을 참조하세요. 모든 인터페이스와 타입은 TypeScript로 완전히 타입이 지정되어 있어 IDE 자동완성으로 확인할 수 있습니다.

### 주요 API 요약

| API                      | 설명                                     |
| ------------------------ | ---------------------------------------- |
| `MathCanvas`             | Vue 3 캔버스 컴포넌트 (마우스/터치 입력) |
| `InferenceEngine`        | ONNX 추론 엔진 (인코더+디코더)           |
| `preprocessStrokes()`    | 스트로크 → 정규화된 텐서 변환            |
| `isStrokeMeaningful()`   | 의미 있는 입력인지 검증 (점/탭 필터링)   |
| `loadVocab()`            | 어휘 JSON 로드 (캐싱됨)                  |
| `decodeTokenIds()`       | 토큰 ID → LaTeX 문자열 변환              |
| `decodeToTokenArray()`   | 토큰 ID → LaTeX 토큰 배열 변환           |
| `repairLatex()`          | LaTeX 자동 수정 (괄호 균형, 인수 수정)   |
| `isCompleteExpression()` | 수식 완결성 검사                         |
| `fetchWithCache()`       | IndexedDB 캐싱 포함 모델 다운로드        |

## 작동 원리

이 라이브러리는 [CoMER](https://github.com/Green-Wood/CoMER) (Coverage-guided Multi-scale Encoder-decoder Transformer, ECCV 2022)를 ONNX Runtime Web으로 브라우저에서 완전히 실행합니다.

### 모델 파이프라인

1. **스트로크 캡처** — 캔버스가 포인터/터치 이벤트를 좌표와 선 두께가 포함된 `Stroke[]`로 수집합니다.
2. **전처리** — [CROHME](https://www.isical.ac.in/~crohme/) 학습 데이터 규격에 따라 스트로크를 오프스크린 캔버스에 렌더링합니다 (검은 배경에 흰 스트로크, 좌상단 정렬). 이미지를 높이 256px로 스케일링하고, 너비는 64px 배수로 동적 조정한 후 그레이스케일 float32 텐서와 바이너리 패딩 마스크로 변환합니다.
3. **인코더** — DenseNet 백본이 다중 스케일 시각 특성을 추출합니다. Transformer 인코더가 위치 임베딩과 함께 컨텍스트 특성 맵을 생성합니다.
4. **디코더** — 커버리지 어텐션을 갖춘 오토리그레시브 Transformer 디코더가 토큰 ID를 하나씩 생성합니다. 길이 정규화가 적용된 빔 서치가 최적의 가설을 선택합니다. 반복 감지가 퇴화 시퀀스를 중단합니다.
5. **토크나이저** — 토큰 ID를 `vocab.json`을 통해 LaTeX 심볼로 매핑합니다 (숫자, 연산자, 그리스 문자, 분수/위첨자 등 구조 토큰 포함 245개 심볼).

### 최적화

- **INT8 양자화** — FP32에서 INT8로 양자화하여 모델 크기를 ~100MB에서 7.2MB로 축소, 정확도 손실 최소화.
- **Web Worker 추론** — ONNX 추론이 별도 Worker 스레드에서 실행되어 1-2초 인식 중 UI 블로킹 방지.
- **동적 입력 너비** — 텐서 너비가 콘텐츠에 맞게 조정되어 패딩에 대한 불필요한 연산을 제거. 단순 수식은 최대 80% 더 빠르게 실행.
- **스트로크 리샘플링** — 원시 터치 포인트를 균일한 3px 간격으로 리샘플링하고 Bézier 곡선으로 렌더링하여 CROHME 학습 데이터 품질과 일치.
- **LaTeX 자동 수정** — `repairLatex()`가 불균형 괄호와 초과 `\frac`/`\sqrt` 인수를 수정. KaTeX 런타임 검증으로 최적 빔 후보 선택.
- **어휘 마스킹** — Number 모드에서 디코더 로짓을 숫자+기본연산자로 제한하여 불가능한 토큰을 빔 서치에서 제거.
- **멀티스레드 WASM** — `SharedArrayBuffer`로 CPU 코어 간 병렬 실행 가능.
- **IndexedDB 캐싱** — 모델을 한 번 다운로드 후 로컬에 캐시하여 즉시 재로드.

## 벤치마크

[CROHME 2014](https://www.isical.ac.in/~crohme/) 테스트셋(986개 손글씨 수학 수식)에서 ink-on 전체 파이프라인을 end-to-end로 평가했습니다.

| 모델                      | ExpRate    | ≤1 edit    | ≤2 edits   | 크기       |
| ------------------------- | ---------- | ---------- | ---------- | ---------- |
| CoMER 논문 (FP32)         | 59.33%     | —          | —          | ~100 MB    |
| **ink-on (INT8, beam=3)** | **36.41%** | **53.25%** | **65.82%** | **7.2 MB** |

ExpRate 차이는 INT8 양자화(92.8% 크기 축소), 전처리 차이, 제한된 어휘(113개 토큰)의 복합 효과입니다. 재현 스크립트는 [benchmark/](./benchmark/)를 참조하세요.

## 라이선스

[Apache License 2.0](./LICENSE) — Copyright 2025 kimseungdae
