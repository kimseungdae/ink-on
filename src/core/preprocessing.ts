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
const W_ALIGN = 64; // align width to multiple of 64 for conv/pool layers
const TARGET_H = 128; // CROHME images avg ~107px height; scale content to this range
const PAD = 16;

let _rawCanvas: HTMLCanvasElement | null = null;
let _targetCanvas: HTMLCanvasElement | null = null;

function createCanvas(
  w: number,
  h: number,
  cache?: "raw" | "target",
): HTMLCanvasElement {
  const fw = Math.max(1, Math.floor(w));
  const fh = Math.max(1, Math.floor(h));
  if (cache) {
    const ref = cache === "raw" ? _rawCanvas : _targetCanvas;
    if (ref && ref.width === fw && ref.height === fh) return ref;
  }
  const c = document.createElement("canvas");
  c.width = fw;
  c.height = fh;
  if (cache === "raw") _rawCanvas = c;
  else if (cache === "target") _targetCanvas = c;
  return c;
}

export function resamplePoints(
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

function computeBBox(strokes: Stroke[]): {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
} {
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

function renderStrokes(strokes: Stroke[]): HTMLCanvasElement {
  const bbox = computeBBox(strokes);
  const rawW = Math.max(1, Math.ceil(bbox.maxX - bbox.minX));
  const rawH = Math.max(1, Math.ceil(bbox.maxY - bbox.minY));
  const canvas = createCanvas(rawW + PAD * 2, rawH + PAD * 2, "raw");
  const ctx = canvas.getContext("2d")!;

  // CROHME convention: black background, white strokes
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

function scaleToFit(src: HTMLCanvasElement): {
  canvas: HTMLCanvasElement;
  contentH: number;
  contentW: number;
  canvasW: number;
} {
  // Scale content to TARGET_H height (matching CROHME training proportions)
  const scale = Math.min(TARGET_H / src.height, MAX_W / src.width);
  const dw = Math.max(1, Math.round(src.width * scale));
  const dh = Math.max(1, Math.round(src.height * scale));

  // Dynamic width: content width + margin, aligned to W_ALIGN, clamped to [MIN_W, MAX_W]
  const canvasW = Math.min(
    MAX_W,
    Math.max(MIN_W, Math.ceil((dw + PAD) / W_ALIGN) * W_ALIGN),
  );

  const target = createCanvas(canvasW, MODEL_H, "target");
  const ctx = target.getContext("2d")!;

  // Black background (matching CROHME training data)
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, canvasW, MODEL_H);

  // Top-left alignment (CROHME training convention, NOT centered)
  ctx.drawImage(src, 0, 0, dw, dh);

  return { canvas: target, contentH: dh, contentW: dw, canvasW };
}

function canvasToGrayscaleTensor(canvas: HTMLCanvasElement): Float32Array {
  const ctx = canvas.getContext("2d")!;
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

const MIN_STROKE_SIZE = 8; // minimum bbox dimension in canvas pixels
const MIN_TOTAL_POINTS = 6; // minimum total points across all strokes
const MIN_PATH_LENGTH = 15; // minimum total path length in canvas pixels

export function isStrokeMeaningful(strokes: Stroke[]): boolean {
  if (strokes.length === 0) return false;
  const bbox = computeBBox(strokes);
  const w = bbox.maxX - bbox.minX;
  const h = bbox.maxY - bbox.minY;
  if (w < MIN_STROKE_SIZE && h < MIN_STROKE_SIZE) return false;

  // Check total points — dots/taps have very few points
  let totalPoints = 0;
  let totalLength = 0;
  for (const s of strokes) {
    totalPoints += s.points.length;
    for (let i = 1; i < s.points.length; i++) {
      const dx = s.points[i]!.x - s.points[i - 1]!.x;
      const dy = s.points[i]!.y - s.points[i - 1]!.y;
      totalLength += Math.sqrt(dx * dx + dy * dy);
    }
  }
  return totalPoints >= MIN_TOTAL_POINTS && totalLength >= MIN_PATH_LENGTH;
}

export function preprocessStrokes(strokes: Stroke[]): PreprocessResult {
  const rawCanvas = renderStrokes(strokes);
  const { canvas, contentH, contentW, canvasW } = scaleToFit(rawCanvas);
  const tensor = canvasToGrayscaleTensor(canvas);

  // Mask: 1 = padding, 0 = valid content area
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
