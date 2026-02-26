<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import MathCanvas from './components/MathCanvas.vue';
import { InferenceEngine, preprocessStrokes, isStrokeMeaningful, loadVocab } from './core';
import type { Stroke, RecognitionResult, Vocab } from './core';

const canvasRef = ref<InstanceType<typeof MathCanvas> | null>(null);
const result = ref<RecognitionResult | null>(null);
const status = ref('Loading model...');
const isReady = ref(false);
const isRecognizing = ref(false);
const loadProgress = ref('');
const lowMemWarning = ref(false);

let engine: InferenceEngine | null = null;
let vocab: Vocab | null = null;
let idleTimer: ReturnType<typeof setTimeout> | null = null;
const IDLE_MS = 1200;

const renderedMath = computed(() => {
  if (!result.value?.latex) return '';
  try {
    return katex.renderToString(result.value.latex, {
      throwOnError: false,
      displayMode: true,
    });
  } catch {
    return '';
  }
});

onMounted(async () => {
  try {
    loadProgress.value = 'Loading vocabulary...';
    vocab = await loadVocab('/models/comer/vocab.json');

    // Adaptive beam width based on device capability
    const cores = navigator.hardwareConcurrency || 2;
    const mem = (navigator as unknown as { deviceMemory?: number }).deviceMemory || 4;
    const beamWidth = mem <= 2 || cores <= 2 ? 1 : cores <= 4 ? 2 : 3;
    if (mem <= 2) lowMemWarning.value = true;

    loadProgress.value = `Loading ONNX models (beam=${beamWidth})...`;
    engine = new InferenceEngine({
      encoderUrl: '/models/comer/encoder_int8.onnx',
      decoderUrl: '/models/comer/decoder_int8.onnx',
      beamWidth,
      executionProvider: 'wasm',
    });
    await engine.init();

    isReady.value = true;
    status.value = 'Draw a math expression';
  } catch (err) {
    status.value = `Failed to load: ${err}`;
  }
});

onUnmounted(() => {
  engine?.dispose();
});

async function onStrokesChange(strokes: Stroke[]) {
  if (!engine || !vocab || strokes.length === 0) {
    result.value = null;
    return;
  }

  // Skip tiny strokes (dots, accidental taps)
  if (!isStrokeMeaningful(strokes)) {
    result.value = null;
    return;
  }

  if (idleTimer) clearTimeout(idleTimer);
  status.value = 'Waiting...';
  idleTimer = setTimeout(async () => {
    isRecognizing.value = true;
    status.value = 'Recognizing...';
    try {
      const input = preprocessStrokes(strokes);
      const res = await engine!.recognize(input, vocab!);
      // Ignore empty or garbage results
      if (!res.latex.trim() || res.tokenIds.length === 0) {
        result.value = null;
        status.value = 'Draw a math expression';
      } else {
        result.value = res;
        status.value = `${res.totalMs}ms (enc ${res.encoderMs}ms + dec ${res.decoderMs}ms)`;
      }
    } catch (err) {
      status.value = `Error: ${err}`;
    } finally {
      isRecognizing.value = false;
    }
  }, IDLE_MS);
}

function handleClear() {
  canvasRef.value?.clear();
  result.value = null;
  status.value = 'Draw a math expression';
}

function handleUndo() {
  canvasRef.value?.undo();
}

function handleCopy() {
  if (result.value?.latex) {
    navigator.clipboard.writeText(result.value.latex);
  }
}
</script>

<template>
  <div class="app">
    <header>
      <h1>Math Handwrite</h1>
      <p class="subtitle">Handwritten math recognition in the browser</p>
    </header>

    <div v-if="lowMemWarning" class="low-mem-warning">
      Low memory device detected — using greedy decode for speed
    </div>

    <div class="canvas-wrap" :class="{ recognizing: isRecognizing }">
      <MathCanvas
        ref="canvasRef"
        :width="700"
        :height="300"
        :line-width="3"
        @strokes-change="onStrokesChange"
      />
      <div v-if="!isReady" class="canvas-overlay">
        <div class="loader" />
        <span>{{ loadProgress }}</span>
      </div>
    </div>

    <div class="toolbar">
      <button @click="handleUndo" :disabled="!isReady">Undo</button>
      <button @click="handleClear" :disabled="!isReady">Clear</button>
      <span class="status" :class="{ active: isRecognizing }">{{ status }}</span>
    </div>

    <transition name="fade">
      <div v-if="result" class="result-panel">
        <div v-if="renderedMath" class="math-render" v-html="renderedMath" />
        <div class="latex-raw">
          <code>{{ result.latex }}</code>
          <button class="copy-btn" @click="handleCopy">Copy</button>
        </div>
      </div>
    </transition>

    <footer>
      <span>CoMER (ECCV 2022) &middot; ONNX Runtime Web &middot; INT8 7.2MB</span>
    </footer>
  </div>
</template>

<style scoped>
.app {
  max-width: 780px;
  margin: 0 auto;
  padding: 1.5rem 1rem;
  font-family: system-ui, -apple-system, sans-serif;
  color: #1a1a1a;
}

header { text-align: center; margin-bottom: 1.25rem; }
h1 { font-size: 1.5rem; margin: 0; }
.subtitle { color: #888; font-size: 0.85rem; margin: 0.25rem 0 0; }

.low-mem-warning {
  padding: 0.4rem 0.75rem;
  margin-bottom: 0.75rem;
  background: #fff8e6;
  border: 1px solid #f0d58c;
  border-radius: 6px;
  font-size: 0.8rem;
  color: #8a6d00;
  text-align: center;
}

.canvas-wrap {
  position: relative;
  border: 2px solid #ddd;
  border-radius: 10px;
  overflow: hidden;
  transition: border-color 0.3s;
}
.canvas-wrap.recognizing { border-color: #4a90d9; }
.canvas-wrap :deep(.math-canvas) { border: none; border-radius: 0; display: block; width: 100%; }

.canvas-overlay {
  position: absolute;
  inset: 0;
  background: rgba(255,255,255,0.92);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  font-size: 0.9rem;
  color: #666;
}

.loader {
  width: 28px; height: 28px;
  border: 3px solid #e0e0e0;
  border-top-color: #4a90d9;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

.toolbar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.75rem;
}

button {
  padding: 0.4rem 1.2rem;
  border: 1px solid #d0d0d0;
  border-radius: 6px;
  background: #fafafa;
  cursor: pointer;
  font-size: 0.85rem;
  transition: background 0.15s;
}
button:hover:not(:disabled) { background: #eee; }
button:disabled { opacity: 0.4; cursor: default; }

.status {
  margin-left: auto;
  font-size: 0.8rem;
  color: #999;
  transition: color 0.3s;
}
.status.active { color: #4a90d9; }

.result-panel {
  margin-top: 1rem;
  border: 1px solid #e8e8e8;
  border-radius: 10px;
  overflow: hidden;
}

.math-render {
  padding: 1.25rem 1rem;
  text-align: center;
  font-size: 1.4rem;
  background: #fff;
  min-height: 3rem;
  display: flex;
  align-items: center;
  justify-content: center;
}

.latex-raw {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.75rem;
  background: #f5f6f8;
  border-top: 1px solid #e8e8e8;
}
.latex-raw code {
  flex: 1;
  font-size: 0.85rem;
  color: #555;
  overflow-x: auto;
  white-space: nowrap;
}
.copy-btn {
  padding: 0.25rem 0.6rem;
  font-size: 0.75rem;
  border: 1px solid #d0d0d0;
  flex-shrink: 0;
}

.fade-enter-active, .fade-leave-active { transition: opacity 0.25s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

footer {
  text-align: center;
  margin-top: 1.5rem;
  font-size: 0.75rem;
  color: #bbb;
}

@media (max-width: 600px) {
  .app { padding: 0.75rem 0.5rem; }
  h1 { font-size: 1.2rem; }
  .subtitle { font-size: 0.75rem; }
  header { margin-bottom: 0.75rem; }
  .canvas-wrap { border-radius: 8px; }
  .toolbar { gap: 0.35rem; margin-top: 0.5rem; }
  button { padding: 0.35rem 0.8rem; font-size: 0.8rem; }
  .status { font-size: 0.7rem; }
  .math-render { padding: 0.75rem 0.5rem; font-size: 1.1rem; }
  .latex-raw { padding: 0.4rem 0.5rem; }
  .latex-raw code { font-size: 0.75rem; }
  .result-panel { border-radius: 8px; }
  footer { margin-top: 1rem; }
}
</style>
