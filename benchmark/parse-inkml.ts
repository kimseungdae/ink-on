import { readFileSync, readdirSync } from "fs";
import { resolve } from "path";
import { XMLParser } from "fast-xml-parser";

export interface StrokePoint {
  x: number;
  y: number;
}

export interface Stroke {
  points: StrokePoint[];
  lineWidth: number;
}

export interface CROHMESample {
  id: string;
  groundTruth: string;
  strokes: Stroke[];
}

const parser = new XMLParser({
  ignoreAttributes: false,
  attributeNamePrefix: "@_",
  isArray: (name) => name === "trace" || name === "annotation",
});

function parseTraceData(traceStr: string): StrokePoint[] {
  return traceStr
    .trim()
    .split(",")
    .map((pair) => {
      const parts = pair.trim().split(/\s+/);
      return { x: parseFloat(parts[0]!), y: parseFloat(parts[1]!) };
    })
    .filter((p) => !isNaN(p.x) && !isNaN(p.y));
}

export function parseInkML(filePath: string): CROHMESample | null {
  const xml = readFileSync(filePath, "utf-8");
  const parsed = parser.parse(xml);
  const ink = parsed.ink;
  if (!ink) return null;

  // Extract ground truth from first annotation with type="truth" that starts with $
  let groundTruth = "";
  const annotations = Array.isArray(ink.annotation)
    ? ink.annotation
    : [ink.annotation];
  for (const ann of annotations) {
    if (ann?.["@_type"] === "truth") {
      const text = typeof ann === "object" ? ann["#text"] : ann;
      if (typeof text === "string" && text.startsWith("$")) {
        groundTruth = text
          .replace(/^\$\s*/, "")
          .replace(/\s*\$$/, "")
          .trim();
        break;
      }
    }
  }

  if (!groundTruth) return null;

  // Extract traces
  const traces = Array.isArray(ink.trace)
    ? ink.trace
    : ink.trace
      ? [ink.trace]
      : [];
  const strokes: Stroke[] = [];

  for (const trace of traces) {
    const data = typeof trace === "object" ? trace["#text"] : trace;
    if (typeof data !== "string") continue;
    const points = parseTraceData(data);
    if (points.length > 0) {
      strokes.push({ points, lineWidth: 3 });
    }
  }

  if (strokes.length === 0) return null;

  const id = filePath.split(/[/\\]/).pop()?.replace(".inkml", "") ?? "";
  return { id, groundTruth, strokes };
}

export function loadCROHMEDataset(dir: string): CROHMESample[] {
  const files = readdirSync(dir).filter((f) => f.endsWith(".inkml"));
  const samples: CROHMESample[] = [];
  let skipped = 0;

  for (const file of files) {
    const sample = parseInkML(resolve(dir, file));
    if (sample) {
      samples.push(sample);
    } else {
      skipped++;
    }
  }

  console.log(`Loaded ${samples.length} samples, skipped ${skipped}`);
  return samples;
}
