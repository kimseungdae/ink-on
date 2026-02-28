import { describe, it, expect } from "vitest";
import { decodeToTokenArray, decodeTokenIds } from "../tokenizer";
import type { Vocab } from "../tokenizer";

const vocab: Vocab = {
  word2idx: { "<pad>": 0, "<sos>": 1, "<eos>": 2, x: 3, "+": 4, y: 5 },
  idx2word: {
    "0": "<pad>",
    "1": "<sos>",
    "2": "<eos>",
    "3": "x",
    "4": "+",
    "5": "y",
  },
  special_tokens: { pad: 0, sos: 1, eos: 2 },
  vocab_size: 6,
};

describe("decodeToTokenArray", () => {
  it("filters out special tokens", () => {
    expect(decodeToTokenArray([1, 3, 4, 5, 2], vocab)).toEqual(["x", "+", "y"]);
  });

  it("returns empty array for only special tokens", () => {
    expect(decodeToTokenArray([1, 2], vocab)).toEqual([]);
  });

  it("returns empty array for empty input", () => {
    expect(decodeToTokenArray([], vocab)).toEqual([]);
  });

  it("skips unknown token ids", () => {
    expect(decodeToTokenArray([3, 99, 5], vocab)).toEqual(["x", "y"]);
  });

  it("filters pad tokens mixed in", () => {
    expect(decodeToTokenArray([0, 3, 0, 5], vocab)).toEqual(["x", "y"]);
  });
});

describe("decodeTokenIds", () => {
  it("returns space-joined string", () => {
    expect(decodeTokenIds([1, 3, 4, 5, 2], vocab)).toBe("x + y");
  });

  it("returns empty string for empty tokens", () => {
    expect(decodeTokenIds([], vocab)).toBe("");
  });
});
