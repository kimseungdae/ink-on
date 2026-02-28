import { describe, it, expect } from "vitest";
import { resamplePoints, isStrokeMeaningful } from "../preprocessing";
import type { Stroke, StrokePoint } from "../preprocessing";

describe("resamplePoints", () => {
  it("returns single point as-is", () => {
    const pts: StrokePoint[] = [{ x: 0, y: 0 }];
    expect(resamplePoints(pts)).toEqual(pts);
  });

  it("returns empty array for empty input", () => {
    expect(resamplePoints([])).toEqual([]);
  });

  it("includes first and last points", () => {
    const pts: StrokePoint[] = [
      { x: 0, y: 0 },
      { x: 10, y: 0 },
      { x: 20, y: 0 },
    ];
    const result = resamplePoints(pts, 3);
    expect(result[0]).toEqual({ x: 0, y: 0 });
    expect(result[result.length - 1]).toEqual({ x: 20, y: 0 });
  });

  it("produces evenly spaced points on a straight line", () => {
    const pts: StrokePoint[] = [
      { x: 0, y: 0 },
      { x: 30, y: 0 },
    ];
    const result = resamplePoints(pts, 10);
    // first(0) + resampled(10, 20, 30) + last(30) = 5 points
    expect(result.length).toBe(5);
    expect(result[0]!.x).toBeCloseTo(0);
    expect(result[1]!.x).toBeCloseTo(10);
    expect(result[2]!.x).toBeCloseTo(20);
    expect(result[3]!.x).toBeCloseTo(30);
    expect(result[4]!.x).toBeCloseTo(30); // duplicate last point
  });

  it("preserves two-point strokes", () => {
    const pts: StrokePoint[] = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
    ];
    const result = resamplePoints(pts, 10);
    // Distance < interval → first + last = 2 points
    expect(result.length).toBe(2);
  });
});

describe("isStrokeMeaningful", () => {
  it("returns false for empty strokes", () => {
    expect(isStrokeMeaningful([])).toBe(false);
  });

  it("returns false for tiny dot", () => {
    const strokes: Stroke[] = [
      {
        points: [
          { x: 10, y: 10 },
          { x: 11, y: 10 },
        ],
        lineWidth: 3,
      },
    ];
    expect(isStrokeMeaningful(strokes)).toBe(false);
  });

  it("returns false for very few points", () => {
    const strokes: Stroke[] = [
      {
        points: [
          { x: 0, y: 0 },
          { x: 50, y: 50 },
        ],
        lineWidth: 3,
      },
    ];
    // Only 2 points, totalPoints < MIN_TOTAL_POINTS (6)
    expect(isStrokeMeaningful(strokes)).toBe(false);
  });

  it("returns true for a substantial stroke", () => {
    const points: StrokePoint[] = [];
    for (let i = 0; i <= 20; i++) {
      points.push({ x: i * 5, y: 0 });
    }
    const strokes: Stroke[] = [{ points, lineWidth: 3 }];
    expect(isStrokeMeaningful(strokes)).toBe(true);
  });

  it("returns true for multiple small strokes meeting thresholds", () => {
    const makeStroke = (x1: number, x2: number): Stroke => ({
      points: Array.from({ length: 5 }, (_, i) => ({
        x: x1 + (x2 - x1) * (i / 4),
        y: 50,
      })),
      lineWidth: 3,
    });
    const strokes = [makeStroke(0, 30), makeStroke(40, 70)];
    expect(isStrokeMeaningful(strokes)).toBe(true);
  });
});
