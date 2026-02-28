import { describe, it, expect } from "vitest";
import { applyModeMask, isStructurallyValid } from "../inference";

describe("applyModeMask", () => {
  it("does nothing when allowed is null", () => {
    const logProbs = new Float64Array([0.1, 0.2, 0.3]);
    applyModeMask(logProbs, 3, null);
    expect(logProbs[0]).toBe(0.1);
    expect(logProbs[1]).toBe(0.2);
    expect(logProbs[2]).toBe(0.3);
  });

  it("masks disallowed tokens to -Infinity", () => {
    const logProbs = new Float64Array([0.1, 0.2, 0.3, 0.4]);
    const allowed = new Set([0, 2]);
    applyModeMask(logProbs, 4, allowed);
    expect(logProbs[0]).toBe(0.1);
    expect(logProbs[1]).toBe(-Infinity);
    expect(logProbs[2]).toBe(0.3);
    expect(logProbs[3]).toBe(-Infinity);
  });

  it("keeps all when all are allowed", () => {
    const logProbs = new Float64Array([0.5, 0.5]);
    applyModeMask(logProbs, 2, new Set([0, 1]));
    expect(logProbs[0]).toBe(0.5);
    expect(logProbs[1]).toBe(0.5);
  });
});

describe("isStructurallyValid", () => {
  it("returns false for empty string", () => {
    expect(isStructurallyValid("")).toBe(false);
  });

  it("returns true for simple expression", () => {
    expect(isStructurallyValid("x + y")).toBe(true);
  });

  it("returns true for balanced braces", () => {
    expect(isStructurallyValid("\\frac { a } { b }")).toBe(true);
  });

  it("returns false for unbalanced open brace", () => {
    expect(isStructurallyValid("\\frac { a { b }")).toBe(false);
  });

  it("returns false for unbalanced close brace", () => {
    expect(isStructurallyValid("a } b")).toBe(false);
  });

  it("returns false for \\frac with only 1 braced group", () => {
    expect(isStructurallyValid("\\frac { a }")).toBe(false);
  });

  it("returns true for \\frac with 2 braced groups", () => {
    expect(isStructurallyValid("\\frac { 1 } { 2 }")).toBe(true);
  });

  it("returns false for \\sqrt with no argument", () => {
    expect(isStructurallyValid("\\sqrt")).toBe(false);
  });

  it("returns true for \\sqrt with argument", () => {
    expect(isStructurallyValid("\\sqrt { x }")).toBe(true);
  });

  it("returns true for nested fractions", () => {
    expect(isStructurallyValid("\\frac { \\frac { a } { b } } { c }")).toBe(
      true,
    );
  });
});
