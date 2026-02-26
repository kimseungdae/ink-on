<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue';
import MathCanvas from './components/MathCanvas.vue';
import { InferenceEngine, preprocessStrokes, loadVocab } from './core';
import type { Stroke, RecognitionResult, Vocab } from './core';

const canvasRef = ref<InstanceType<typeof MathCanvas> | null>(null);
const result = ref<RecognitionResult | null>(null);
const status = ref('Loading model...');
const isRecognizing = ref(false);

let engine: InferenceEngine | null = null;
let vocab: Vocab | null = null;
let idleTimer: ReturnType<typeof setTimeout> | null = null;
const IDLE_MS = 1200; // wait 1.2s of no drawing before recognizing

onMounted(async () => {
  try {
    engine = new InferenceEngine({
      encoderUrl: '/models/comer/encoder_int8.onnx',
      decoderUrl: '/models/comer/decoder_int8.onnx',
      executionProvider: 'wasm',
    });
    vocab = await loadVocab('/models/comer/vocab.json');
    await engine.init();
    status.value = 'Ready - draw a math expression!';
  } catch (err) {
    status.value = `Error: ${err}`;
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

  // Reset idle timer on every stroke end — only recognize after IDLE_MS of no activity
  if (idleTimer) clearTimeout(idleTimer);
  status.value = 'Waiting for input...';
  idleTimer = setTimeout(async () => {
    isRecognizing.value = true;
    status.value = 'Recognizing...';
    try {
      const input = preprocessStrokes(strokes);
      const res = await engine!.recognize(input, vocab!);
      result.value = res;
      status.value = `Done in ${res.totalMs}ms (encoder: ${res.encoderMs}ms, decoder: ${res.decoderMs}ms)`;
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
  status.value = 'Ready - draw a math expression!';
}

function handleUndo() {
  canvasRef.value?.undo();
}
</script>

<template>
  <div class="app">
    <h1>Handwritten Math Recognition</h1>
    <p class="subtitle">CoMER (ECCV 2022) - Browser ONNX Runtime</p>

    <MathCanvas
      ref="canvasRef"
      :width="700"
      :height="300"
      :line-width="3"
      @strokes-change="onStrokesChange"
    />

    <div class="controls">
      <button @click="handleUndo">Undo</button>
      <button @click="handleClear">Clear</button>
    </div>

    <div class="status" :class="{ recognizing: isRecognizing }">
      {{ status }}
    </div>

    <div v-if="result" class="result">
      <div class="latex-output">
        <strong>LaTeX:</strong>
        <code>{{ result.latex }}</code>
      </div>
    </div>
  </div>
</template>

<style scoped>
.app {
  max-width: 800px;
  margin: 0 auto;
  padding: 2rem;
  text-align: center;
  font-family: system-ui, sans-serif;
}

h1 {
  margin-bottom: 0.25rem;
}

.subtitle {
  color: #888;
  margin-bottom: 1.5rem;
}

.controls {
  margin-top: 1rem;
  display: flex;
  gap: 0.5rem;
  justify-content: center;
}

button {
  padding: 0.5rem 1.5rem;
  border: 1px solid #ccc;
  border-radius: 6px;
  background: #f5f5f5;
  cursor: pointer;
  font-size: 0.9rem;
}

button:hover {
  background: #e8e8e8;
}

.status {
  margin-top: 1rem;
  color: #666;
  font-size: 0.85rem;
}

.status.recognizing {
  color: #4a90d9;
}

.result {
  margin-top: 1.5rem;
  padding: 1rem;
  background: #f8f9fa;
  border-radius: 8px;
  text-align: left;
}

.latex-output code {
  display: inline-block;
  margin-left: 0.5rem;
  padding: 0.25rem 0.5rem;
  background: #e9ecef;
  border-radius: 4px;
  font-size: 1.1rem;
}
</style>
