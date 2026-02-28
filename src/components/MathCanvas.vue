<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted } from 'vue';
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
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  if ('touches' in e) {
    const touch = e.touches[0]!;
    return {
      x: (touch.clientX - rect.left) * scaleX,
      y: (touch.clientY - rect.top) * scaleY,
    };
  }
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY,
  };
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

function drawSmoothStroke(ctx: CanvasRenderingContext2D, points: StrokePoint[]) {
  if (points.length === 0) return;
  if (points.length === 1) {
    ctx.beginPath();
    ctx.arc(points[0]!.x, points[0]!.y, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fill();
    return;
  }

  ctx.beginPath();
  ctx.moveTo(points[0]!.x, points[0]!.y);

  if (points.length === 2) {
    ctx.lineTo(points[1]!.x, points[1]!.y);
  } else {
    // Quadratic curve through midpoints for smooth strokes
    for (let i = 1; i < points.length - 1; i++) {
      const mx = (points[i]!.x + points[i + 1]!.x) / 2;
      const my = (points[i]!.y + points[i + 1]!.y) / 2;
      ctx.quadraticCurveTo(points[i]!.x, points[i]!.y, mx, my);
    }
    const last = points[points.length - 1]!;
    ctx.lineTo(last.x, last.y);
  }
  ctx.stroke();
}

function redraw() {
  const canvas = canvasRef.value;
  if (!canvas) return;
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const allStrokes = [
    ...strokes.value,
    ...(currentPoints.length > 0
      ? [{ points: currentPoints, lineWidth: props.lineWidth }]
      : []),
  ];

  for (const stroke of allStrokes) {
    ctx.strokeStyle = props.strokeColor;
    ctx.fillStyle = props.strokeColor;
    ctx.lineWidth = stroke.lineWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    drawSmoothStroke(ctx, stroke.points);
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

watch(() => [props.width, props.height], () => {
  redraw();
});

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
  cursor: crosshair;
  touch-action: none;
  background: #fff;
  width: 100%;
  height: auto;
  min-height: 200px;
}
</style>
