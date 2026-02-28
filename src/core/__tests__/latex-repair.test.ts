import { describe, it, expect } from "vitest";
import { repairLatex, isCompleteExpression } from "../latex-repair";

describe("repairLatex", () => {
  it("passes through balanced tokens", () => {
    const tokens = ["\\frac", "{", "a", "}", "{", "b", "}"];
    expect(repairLatex(tokens)).toEqual(tokens);
  });

  it("closes unclosed braces", () => {
    expect(repairLatex(["{", "a"])).toEqual(["{", "a", "}"]);
  });

  it("removes unmatched closing braces", () => {
    expect(repairLatex(["a", "}"])).toEqual(["a"]);
  });

  it("handles nested braces", () => {
    const tokens = ["{", "{", "a", "}", "}"];
    expect(repairLatex(tokens)).toEqual(tokens);
  });

  it("fixes \\frac with 3 braced groups → keeps only 2", () => {
    const tokens = ["\\frac", "{", "a", "}", "{", "b", "}", "{", "c", "}"];
    expect(repairLatex(tokens)).toEqual([
      "\\frac",
      "{",
      "a",
      "}",
      "{",
      "b",
      "}",
    ]);
  });

  it("fixes \\sqrt with 2 braced groups → keeps only 1", () => {
    const tokens = ["\\sqrt", "{", "x", "}", "{", "y", "}"];
    expect(repairLatex(tokens)).toEqual(["\\sqrt", "{", "x", "}"]);
  });

  it("passes through \\frac with exactly 2 args", () => {
    const tokens = ["\\frac", "{", "1", "}", "{", "2", "}"];
    expect(repairLatex(tokens)).toEqual(tokens);
  });

  it("passes through \\sqrt with exactly 1 arg", () => {
    const tokens = ["\\sqrt", "{", "x", "}"];
    expect(repairLatex(tokens)).toEqual(tokens);
  });

  it("handles \\frac with single-token args", () => {
    const tokens = ["\\frac", "a", "b"];
    expect(repairLatex(tokens)).toEqual(["\\frac", "a", "b"]);
  });

  it("returns empty array for empty input", () => {
    expect(repairLatex([])).toEqual([]);
  });

  it("handles multiple \\frac in sequence", () => {
    const tokens = [
      "\\frac",
      "{",
      "1",
      "}",
      "{",
      "2",
      "}",
      "+",
      "\\frac",
      "{",
      "3",
      "}",
      "{",
      "4",
      "}",
    ];
    expect(repairLatex(tokens)).toEqual(tokens);
  });
});

describe("isCompleteExpression", () => {
  it("returns false for empty tokens", () => {
    expect(isCompleteExpression([])).toBe(false);
  });

  it("returns false for lone \\sum", () => {
    expect(isCompleteExpression(["\\sum"])).toBe(false);
  });

  it("returns false for lone \\frac", () => {
    expect(isCompleteExpression(["\\frac"])).toBe(false);
  });

  it("returns false for lone \\sqrt", () => {
    expect(isCompleteExpression(["\\sqrt"])).toBe(false);
  });

  it("returns true for simple number", () => {
    expect(isCompleteExpression(["3"])).toBe(true);
  });

  it("returns true for complete fraction", () => {
    expect(isCompleteExpression(["\\frac", "{", "1", "}", "{", "2", "}"])).toBe(
      true,
    );
  });

  it("returns true for expression with operators", () => {
    expect(isCompleteExpression(["x", "+", "y"])).toBe(true);
  });

  it("returns false for \\frac with only 1 arg", () => {
    expect(isCompleteExpression(["\\frac", "{", "1", "}"])).toBe(false);
  });

  it("returns false for \\sqrt with no arg", () => {
    expect(isCompleteExpression(["\\sqrt"])).toBe(false);
  });

  it("returns true for \\sqrt with 1 arg", () => {
    expect(isCompleteExpression(["\\sqrt", "{", "x", "}"])).toBe(true);
  });
});
