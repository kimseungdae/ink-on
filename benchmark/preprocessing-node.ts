import { createCanvas, Canvas, CanvasRenderingContext2D } from "canvas";

export interface StrokePoint {
  x: number;
  y: number;
}

export interface Stroke {
  points: StrokePoint[];
  lineWidth: number;
}

export interface PreprocessResult {
  tensor: Float32Array;
  height: number;
  width: number;
  mask: Uint8Array;
  maskHeight: number;
  maskWidth: number;
}

const MODEL_H = 256;
const MAX_W = 1024;
const MIN_W = 128;
const W_ALIGN = 64;
const TARGET_H = 128;
const PAD = 16;

function resamplePoints(
  points: StrokePoint[],
  interval: number = 3,
): StrokePoint[] {
  if (points.length < 2) return points;
  const resampled: StrokePoint[] = [points[0]!];
  let remaining = interval;

  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1]!;
    const curr = points[i]!;
    const dx = curr.x - prev.x;
    const dy = curr.y - prev.y;
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist <= remaining) {
      remaining -= dist;
      continue;
    }
    let covered = remaining;
    while (covered <= dist) {
      const t = covered / dist;
      resampled.push({
        x: prev.x + dx * t,
        y: prev.y + dy * t,
      });
      covered += interval;
    }
    remaining = covered - dist;
  }
  resampled.push(points[points.length - 1]!);
  return resampled;
}

function computeBBox(strokes: Stroke[]) {
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  for (const s of strokes) {
    for (const p of s.points) {
      if (p.x < minX) minX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }
  }
  return { minX, minY, maxX, maxY };
}

function renderStrokes(strokes: Stroke[]): Canvas {
  const bbox = computeBBox(strokes);
  const rawW = Math.max(1, Math.ceil(bbox.maxX - bbox.minX));
  const rawH = Math.max(1, Math.ceil(bbox.maxY - bbox.minY));
  const canvas = createCanvas(rawW + PAD * 2, rawH + PAD * 2);
  const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;

  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#ffffff";
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  for (const stroke of strokes) {
    if (stroke.points.length === 0) continue;
    const pts = resamplePoints(stroke.points);
    ctx.beginPath();
    ctx.lineWidth = Math.max(2, stroke.lineWidth);
    const first = pts[0]!;
    ctx.moveTo(first.x - bbox.minX + PAD, first.y - bbox.minY + PAD);

    if (pts.length === 2) {
      ctx.lineTo(pts[1]!.x - bbox.minX + PAD, pts[1]!.y - bbox.minY + PAD);
    } else if (pts.length > 2) {
      for (let i = 1; i < pts.length - 1; i++) {
        const mx = (pts[i]!.x + pts[i + 1]!.x) / 2 - bbox.minX + PAD;
        const my = (pts[i]!.y + pts[i + 1]!.y) / 2 - bbox.minY + PAD;
        ctx.quadraticCurveTo(
          pts[i]!.x - bbox.minX + PAD,
          pts[i]!.y - bbox.minY + PAD,
          mx,
          my,
        );
      }
      const last = pts[pts.length - 1]!;
      ctx.lineTo(last.x - bbox.minX + PAD, last.y - bbox.minY + PAD);
    }
    ctx.stroke();
  }
  return canvas;
}

function scaleToFit(src: Canvas) {
  const scale = Math.min(TARGET_H / src.height, MAX_W / src.width);
  const dw = Math.max(1, Math.round(src.width * scale));
  const dh = Math.max(1, Math.round(src.height * scale));
  const canvasW = Math.min(
    MAX_W,
    Math.max(MIN_W, Math.ceil((dw + PAD) / W_ALIGN) * W_ALIGN),
  );

  const target = createCanvas(canvasW, MODEL_H);
  const ctx = target.getContext("2d") as CanvasRenderingContext2D;

  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, canvasW, MODEL_H);
  ctx.drawImage(src as any, 0, 0, dw, dh);

  return { canvas: target, contentH: dh, contentW: dw, canvasW };
}

function canvasToGrayscaleTensor(canvas: Canvas): Float32Array {
  const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const { data } = imageData;
  const pixels = canvas.width * canvas.height;
  const tensor = new Float32Array(pixels);

  for (let i = 0; i < pixels; i++) {
    const offset = i * 4;
    const gray =
      (data[offset]! * 0.299 +
        data[offset + 1]! * 0.587 +
        data[offset + 2]! * 0.114) /
      255;
    tensor[i] = gray;
  }
  return tensor;
}

export function preprocessStrokes(strokes: Stroke[]): PreprocessResult {
  const rawCanvas = renderStrokes(strokes);
  const { canvas, contentH, contentW, canvasW } = scaleToFit(rawCanvas);
  const tensor = canvasToGrayscaleTensor(canvas);

  const mask = new Uint8Array(MODEL_H * canvasW);
  for (let y = 0; y < MODEL_H; y++) {
    for (let x = 0; x < canvasW; x++) {
      mask[y * canvasW + x] = y < contentH && x < contentW ? 0 : 1;
    }
  }

  return {
    tensor,
    height: MODEL_H,
    width: canvasW,
    mask,
    maskHeight: MODEL_H,
    maskWidth: canvasW,
  };
}
