<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue';
import type { Stroke, StrokePoint } from '../core/preprocessing';

const props = withDefaults(defineProps<{
  width?: number;
  height?: number;
  lineWidth?: number;
  strokeColor?: string;
}>(), {
  width: 600,
  height: 300,
  lineWidth: 3,
  strokeColor: '#000000',
});

const emit = defineEmits<{
  (e: 'strokesChange', strokes: Stroke[]): void;
}>();

const canvasRef = ref<HTMLCanvasElement | null>(null);
const strokes = ref<Stroke[]>([]);
const isDrawing = ref(false);
let currentPoints: StrokePoint[] = [];

function getPos(e: MouseEvent | TouchEvent): StrokePoint {
  const canvas = canvasRef.value!;
  const rect = canvas.getBoundingClientRect();
  if ('touches' in e) {
    const touch = e.touches[0];
    return { x: touch.clientX - rect.left, y: touch.clientY - rect.top };
  }
  return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

function startDraw(e: MouseEvent | TouchEvent) {
  e.preventDefault();
  isDrawing.value = true;
  currentPoints = [getPos(e)];
}

function draw(e: MouseEvent | TouchEvent) {
  if (!isDrawing.value) return;
  e.preventDefault();
  const pos = getPos(e);
  currentPoints.push(pos);
  redraw();
}

function endDraw() {
  if (!isDrawing.value) return;
  isDrawing.value = false;
  if (currentPoints.length > 0) {
    strokes.value = [...strokes.value, { points: currentPoints, lineWidth: props.lineWidth }];
    currentPoints = [];
    emit('strokesChange', strokes.value);
  }
}

function redraw() {
  const canvas = canvasRef.value;
  if (!canvas) return;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const allStrokes = [...strokes.value, ...(currentPoints.length > 0 ? [{ points: currentPoints, lineWidth: props.lineWidth }] : [])];

  for (const stroke of allStrokes) {
    if (stroke.points.length === 0) continue;
    ctx.beginPath();
    ctx.strokeStyle = props.strokeColor;
    ctx.lineWidth = stroke.lineWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
    for (let i = 1; i < stroke.points.length; i++) {
      ctx.lineTo(stroke.points[i].x, stroke.points[i].y);
    }
    ctx.stroke();
  }
}

function clear() {
  strokes.value = [];
  currentPoints = [];
  redraw();
  emit('strokesChange', []);
}

function undo() {
  if (strokes.value.length === 0) return;
  strokes.value = strokes.value.slice(0, -1);
  redraw();
  emit('strokesChange', strokes.value);
}

defineExpose({ clear, undo, strokes });

onMounted(() => {
  const canvas = canvasRef.value;
  if (!canvas) return;
  canvas.addEventListener('touchstart', startDraw, { passive: false });
  canvas.addEventListener('touchmove', draw, { passive: false });
  canvas.addEventListener('touchend', endDraw);
});

onUnmounted(() => {
  const canvas = canvasRef.value;
  if (!canvas) return;
  canvas.removeEventListener('touchstart', startDraw);
  canvas.removeEventListener('touchmove', draw);
  canvas.removeEventListener('touchend', endDraw);
});
</script>

<template>
  <canvas
    ref="canvasRef"
    :width="width"
    :height="height"
    class="math-canvas"
    @mousedown="startDraw"
    @mousemove="draw"
    @mouseup="endDraw"
    @mouseleave="endDraw"
  />
</template>

<style scoped>
.math-canvas {
  border: 2px solid #ccc;
  border-radius: 8px;
  cursor: crosshair;
  touch-action: none;
  background: #fff;
}
</style>
