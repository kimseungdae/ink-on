import * as ort from "onnxruntime-node";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { resolve } from "path";
import { loadCROHMEDataset, type CROHMESample } from "./parse-inkml.js";
import {
  preprocessStrokes,
  type PreprocessResult,
} from "./preprocessing-node.js";

// --- Vocab ---
interface Vocab {
  word2idx: Record<string, number>;
  idx2word: Record<string, string>;
  special_tokens: { pad: number; sos: number; eos: number };
  vocab_size: number;
}

function loadVocab(path: string): Vocab {
  return JSON.parse(readFileSync(path, "utf-8"));
}

function decodeTokenIds(ids: number[], vocab: Vocab): string {
  return decodeToTokenArray(ids, vocab).join(" ");
}

function decodeToTokenArray(ids: number[], vocab: Vocab): string[] {
  const { sos, eos, pad } = vocab.special_tokens;
  const skip = new Set([sos, eos, pad]);
  const words: string[] = [];
  for (const id of ids) {
    if (skip.has(id)) continue;
    const w = vocab.idx2word[String(id)];
    if (w !== undefined) words.push(w);
  }
  return words;
}

function repairLatex(tokens: string[]): string[] {
  let result = [...tokens];
  result = _balanceBraces(result);
  result = _fixFracArgs(result);
  result = _fixSqrtArgs(result);
  return result;
}

function _balanceBraces(tokens: string[]): string[] {
  const result: string[] = [];
  let depth = 0;
  for (const t of tokens) {
    if (t === "{") {
      depth++;
      result.push(t);
    } else if (t === "}") {
      if (depth > 0) {
        depth--;
        result.push(t);
      }
    } else {
      result.push(t);
    }
  }
  while (depth > 0) {
    result.push("}");
    depth--;
  }
  return result;
}

function _fixFracArgs(tokens: string[]): string[] {
  const result: string[] = [];
  let i = 0;
  while (i < tokens.length) {
    if (tokens[i] === "\\frac") {
      result.push(tokens[i]!);
      i++;
      let groups = 0;
      while (i < tokens.length && groups < 2) {
        if (tokens[i] === "{") {
          groups++;
          let d = 1;
          result.push(tokens[i]!);
          i++;
          while (i < tokens.length && d > 0) {
            if (tokens[i] === "{") d++;
            else if (tokens[i] === "}") d--;
            result.push(tokens[i]!);
            i++;
          }
        } else {
          result.push(tokens[i]!);
          i++;
          groups++;
        }
      }
      while (i < tokens.length && tokens[i] === "{") {
        let d = 1;
        i++;
        while (i < tokens.length && d > 0) {
          if (tokens[i] === "{") d++;
          else if (tokens[i] === "}") d--;
          i++;
        }
      }
    } else {
      result.push(tokens[i]!);
      i++;
    }
  }
  return result;
}

function _fixSqrtArgs(tokens: string[]): string[] {
  const result: string[] = [];
  let i = 0;
  while (i < tokens.length) {
    if (tokens[i] === "\\sqrt") {
      result.push(tokens[i]!);
      i++;
      let groups = 0;
      while (i < tokens.length && groups < 1) {
        if (tokens[i] === "{") {
          groups++;
          let d = 1;
          result.push(tokens[i]!);
          i++;
          while (i < tokens.length && d > 0) {
            if (tokens[i] === "{") d++;
            else if (tokens[i] === "}") d--;
            result.push(tokens[i]!);
            i++;
          }
        } else {
          result.push(tokens[i]!);
          i++;
          groups++;
        }
      }
      while (i < tokens.length && tokens[i] === "{") {
        let d = 1;
        i++;
        while (i < tokens.length && d > 0) {
          if (tokens[i] === "{") d++;
          else if (tokens[i] === "}") d--;
          i++;
        }
      }
    } else {
      result.push(tokens[i]!);
      i++;
    }
  }
  return result;
}

// --- LaTeX Tokenizer (for GT normalization) ---
const LATEX_COMMANDS = [
  "\\rightarrow",
  "\\exists",
  "\\forall",
  "\\lambda",
  "\\limits",
  "\\cdots",
  "\\ldots",
  "\\Delta",
  "\\alpha",
  "\\beta",
  "\\gamma",
  "\\theta",
  "\\sigma",
  "\\prime",
  "\\infty",
  "\\times",
  "\\sqrt",
  "\\frac",
  "\\cdot",
  "\\cos",
  "\\div",
  "\\geq",
  "\\leq",
  "\\lim",
  "\\log",
  "\\neq",
  "\\phi",
  "\\sin",
  "\\sum",
  "\\tan",
  "\\int",
  "\\mu",
  "\\pi",
  "\\pm",
  "\\Pi",
  "\\in",
  "\\{",
  "\\}",
];

function tokenizeLatex(latex: string, vocabKeys: Set<string>): string[] {
  const tokens: string[] = [];
  let i = 0;
  while (i < latex.length) {
    if (latex[i] === " ") {
      i++;
      continue;
    }

    if (latex[i] === "\\") {
      let found = false;
      for (const cmd of LATEX_COMMANDS) {
        if (latex.startsWith(cmd, i)) {
          tokens.push(cmd);
          i += cmd.length;
          found = true;
          break;
        }
      }
      if (!found) {
        // Unknown command - consume until non-alpha
        let j = i + 1;
        while (j < latex.length && /[a-zA-Z]/.test(latex[j]!)) j++;
        tokens.push(latex.slice(i, j));
        i = j;
      }
    } else {
      tokens.push(latex[i]!);
      i++;
    }
  }
  return tokens;
}

function normalizeLatex(latex: string, vocabKeys: Set<string>): string {
  const tokens = tokenizeLatex(latex, vocabKeys);
  return normalizeTokens(tokens).join(" ");
}

// Normalize brace differences: `_ k` → `_ { k }`, `^ 2` → `^ { 2 }`
// Also: `\frac a b` → `\frac { a } { b }`, `\sqrt x` → `\sqrt { x }`
// Remove `\!`, `\,`, `\ ` (LaTeX spacing commands not in vocab)
function normalizeTokens(tokens: string[]): string[] {
  // Remove LaTeX spacing commands
  const filtered = tokens.filter(
    (t) => t !== "\\!" && t !== "\\," && t !== "\\ " && t !== "\\;",
  );

  const result: string[] = [];
  let i = 0;
  while (i < filtered.length) {
    const t = filtered[i]!;

    if (t === "^" || t === "_") {
      result.push(t);
      i++;
      if (i < filtered.length && filtered[i] !== "{") {
        // Single token without braces → wrap in braces
        result.push("{", filtered[i]!, "}");
        i++;
      }
    } else if (t === "\\frac" || t === "\\sqrt") {
      result.push(t);
      i++;
      // For \frac: expect 2 args, for \sqrt: expect 1 arg
      const argCount = t === "\\frac" ? 2 : 1;
      for (let a = 0; a < argCount && i < filtered.length; a++) {
        if (filtered[i] === "{") {
          // Already braced - pass through
          while (i < filtered.length) {
            result.push(filtered[i]!);
            if (filtered[i] === "}") {
              i++;
              break;
            }
            i++;
          }
        } else {
          // Single token without braces → wrap
          result.push("{", filtered[i]!, "}");
          i++;
        }
      }
    } else {
      result.push(t);
      i++;
    }
  }
  return result;
}

function normalizePrediction(predicted: string): string {
  const tokens = predicted.split(" ").filter(Boolean);
  return normalizeTokens(tokens).join(" ");
}

// --- Inference Engine (Node.js) ---
const REPEAT_LIMIT = 3;
const MAX_DECODE_STEPS = 50;

async function runEncoder(
  session: ort.InferenceSession,
  input: PreprocessResult,
): Promise<{ features: ort.Tensor; mask: ort.Tensor }> {
  const pixelValues = new ort.Tensor("float32", input.tensor, [
    1,
    1,
    input.height,
    input.width,
  ]);
  const pixelMask = new ort.Tensor("bool", input.mask, [
    1,
    input.maskHeight,
    input.maskWidth,
  ]);
  const result = await session.run({
    pixel_values: pixelValues,
    pixel_mask: pixelMask,
  });
  return {
    features: result["encoder_features"]!,
    mask: result["encoder_mask"]!,
  };
}

async function runDecoder(
  session: ort.InferenceSession,
  encoderFeatures: ort.Tensor,
  encoderMask: ort.Tensor,
  ids: number[],
): Promise<Float32Array> {
  const inputIds = new ort.Tensor(
    "int64",
    BigInt64Array.from(ids.map(BigInt)),
    [1, ids.length],
  );
  const res = await session.run({
    encoder_features: encoderFeatures,
    encoder_mask: encoderMask,
    input_ids: inputIds,
  });
  return res["logits"]!.data as Float32Array;
}

async function greedyDecode(
  decoderSession: ort.InferenceSession,
  encoderFeatures: ort.Tensor,
  encoderMask: ort.Tensor,
  vocab: Vocab,
): Promise<number[]> {
  const { sos, eos } = vocab.special_tokens;
  const tokenIds: number[] = [sos];
  let repeatCount = 0;
  let lastToken = -1;

  for (let step = 0; step < MAX_DECODE_STEPS; step++) {
    const logits = await runDecoder(
      decoderSession,
      encoderFeatures,
      encoderMask,
      tokenIds,
    );
    const offset = (tokenIds.length - 1) * vocab.vocab_size;
    let maxVal = -Infinity;
    let maxIdx = 0;
    for (let i = 0; i < vocab.vocab_size; i++) {
      const v = logits[offset + i]!;
      if (v > maxVal) {
        maxVal = v;
        maxIdx = i;
      }
    }
    if (maxIdx === eos) break;
    if (maxIdx === lastToken) {
      repeatCount++;
      if (repeatCount >= REPEAT_LIMIT) break;
    } else {
      repeatCount = 0;
    }
    lastToken = maxIdx;
    tokenIds.push(maxIdx);
  }
  return tokenIds.slice(1);
}

async function beamDecode(
  decoderSession: ort.InferenceSession,
  encoderFeatures: ort.Tensor,
  encoderMask: ort.Tensor,
  vocab: Vocab,
  beamWidth: number,
): Promise<number[]> {
  const { sos, eos } = vocab.special_tokens;

  interface Beam {
    logProb: number;
    ids: number[];
    finished: boolean;
  }
  let beams: Beam[] = [{ logProb: 0, ids: [sos], finished: false }];

  for (let step = 0; step < MAX_DECODE_STEPS; step++) {
    const candidates: Beam[] = [];
    for (const beam of beams) {
      if (beam.finished) {
        candidates.push(beam);
        continue;
      }
      const logits = await runDecoder(
        decoderSession,
        encoderFeatures,
        encoderMask,
        beam.ids,
      );
      const offset = (beam.ids.length - 1) * vocab.vocab_size;

      // Log-softmax
      let max = -Infinity;
      for (let i = 0; i < vocab.vocab_size; i++) {
        const v = logits[offset + i]!;
        if (v > max) max = v;
      }
      let sumExp = 0;
      for (let i = 0; i < vocab.vocab_size; i++)
        sumExp += Math.exp(logits[offset + i]! - max);
      const logSumExp = Math.log(sumExp);

      // Top-k
      const logProbs = new Float64Array(vocab.vocab_size);
      for (let i = 0; i < vocab.vocab_size; i++) {
        logProbs[i] = logits[offset + i]! - max - logSumExp;
      }
      const indices = Array.from({ length: vocab.vocab_size }, (_, i) => i);
      indices.sort((a, b) => (logProbs[b] ?? 0) - (logProbs[a] ?? 0));
      const topIndices = indices.slice(0, beamWidth * 2);

      for (const idx of topIndices) {
        const newLogProb = beam.logProb + (logProbs[idx] ?? 0);
        if (idx === eos) {
          candidates.push({
            logProb: newLogProb,
            ids: beam.ids,
            finished: true,
          });
        } else {
          let rptCount = 0;
          for (let j = beam.ids.length - 1; j >= 1; j--) {
            if (beam.ids[j] === idx) rptCount++;
            else break;
          }
          if (rptCount >= REPEAT_LIMIT) {
            candidates.push({
              logProb: newLogProb,
              ids: beam.ids,
              finished: true,
            });
          } else {
            candidates.push({
              logProb: newLogProb,
              ids: [...beam.ids, idx],
              finished: false,
            });
          }
        }
      }
    }
    candidates.sort(
      (a, b) =>
        b.logProb / Math.max(b.ids.length, 1) -
        a.logProb / Math.max(a.ids.length, 1),
    );
    beams = candidates.slice(0, beamWidth);
    if (beams.every((b) => b.finished)) break;
  }

  const completed = beams.filter((b) => b.finished);
  const best =
    completed.length > 0
      ? completed.sort(
          (a, b) =>
            b.logProb / Math.max(b.ids.length, 1) -
            a.logProb / Math.max(a.ids.length, 1),
        )[0]!
      : beams[0]!;
  return best.ids.slice(1);
}

// --- Edit Distance ---
function editDistance(a: string[], b: string[]): number {
  const m = a.length,
    n = b.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () =>
    new Array(n + 1).fill(0),
  );
  for (let i = 0; i <= m; i++) dp[i]![0] = i;
  for (let j = 0; j <= n; j++) dp[0]![j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i]![j] =
        a[i - 1] === b[j - 1]
          ? dp[i - 1]![j - 1]!
          : 1 + Math.min(dp[i - 1]![j]!, dp[i]![j - 1]!, dp[i - 1]![j - 1]!);
    }
  }
  return dp[m]![n]!;
}

// --- Main ---
async function main() {
  const rootDir = resolve(import.meta.dirname!, "..");
  const modelsDir = resolve(rootDir, "public/models/comer");

  // Load vocab
  const vocab = loadVocab(resolve(modelsDir, "vocab.json"));
  const vocabKeys = new Set(Object.keys(vocab.word2idx));
  console.log(`Vocab: ${vocab.vocab_size} tokens`);

  // Load CROHME 2014 test set (has GT)
  const dataDir = resolve(
    import.meta.dirname!,
    "data/Task1_and_Task2/Task1_and_Task2/Task1_onlineRec/MainTask_formula/valid/valid/TestEM2014GT_INKMLs",
  );

  if (!existsSync(dataDir)) {
    console.error("CROHME data not found. Run: npx tsx download-crohme.ts");
    process.exit(1);
  }

  const samples = loadCROHMEDataset(dataDir);
  console.log(`Total samples: ${samples.length}`);

  // Load ONNX sessions
  console.log("Loading ONNX models...");
  const encSession = await ort.InferenceSession.create(
    resolve(modelsDir, "encoder_int8.onnx"),
    { executionProviders: ["cpu"] },
  );
  const decSession = await ort.InferenceSession.create(
    resolve(modelsDir, "decoder_int8.onnx"),
    { executionProviders: ["cpu"] },
  );
  console.log("Models loaded.");

  // Evaluate
  const beamWidth = 3;
  let correct = 0;
  let edit1 = 0;
  let edit2 = 0;
  let vocabMismatch = 0;
  const errors: { id: string; gt: string; pred: string; dist: number }[] = [];
  const total = samples.length;

  const t0 = Date.now();

  for (let i = 0; i < total; i++) {
    const sample = samples[i]!;
    const pct = (((i + 1) / total) * 100).toFixed(1);

    try {
      // Preprocess
      const input = preprocessStrokes(sample.strokes);

      // Encode
      const { features, mask } = await runEncoder(encSession, input);

      // Decode
      const tokenIds =
        beamWidth <= 1
          ? await greedyDecode(decSession, features, mask, vocab)
          : await beamDecode(decSession, features, mask, vocab, beamWidth);

      const tokens = decodeToTokenArray(tokenIds, vocab);
      const repaired = repairLatex(tokens);
      const predicted = repaired.join(" ");
      const gtNorm = normalizeLatex(sample.groundTruth, vocabKeys);
      const predNorm = normalizePrediction(predicted);

      // Check vocab coverage
      const gtTokens = tokenizeLatex(sample.groundTruth, vocabKeys);
      const hasUnknown = gtTokens.some((t) => !vocabKeys.has(t));
      if (hasUnknown) vocabMismatch++;

      // Compare (both normalized)
      const predTokens = predNorm.split(" ").filter(Boolean);
      const gtTokenList = gtNorm.split(" ").filter(Boolean);
      const dist = editDistance(predTokens, gtTokenList);

      if (dist === 0) correct++;
      if (dist <= 1) edit1++;
      if (dist <= 2) edit2++;

      if (dist > 0) {
        errors.push({ id: sample.id, gt: gtNorm, pred: predNorm, dist });
      }

      if ((i + 1) % 50 === 0 || i + 1 === total) {
        const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
        const expRate = ((correct / (i + 1)) * 100).toFixed(2);
        console.log(
          `[${pct}%] ${i + 1}/${total} | ExpRate: ${expRate}% | Elapsed: ${elapsed}s`,
        );
      }
    } catch (err) {
      console.error(`Error on ${sample.id}: ${(err as Error).message}`);
      errors.push({
        id: sample.id,
        gt: normalizeLatex(sample.groundTruth, vocabKeys),
        pred: "ERROR",
        dist: 999,
      });
    }
  }

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

  // Results
  const results = {
    dataset: "CROHME 2014 Test",
    totalSamples: total,
    beamWidth,
    expRate: ((correct / total) * 100).toFixed(2),
    expRate1: ((edit1 / total) * 100).toFixed(2),
    expRate2: ((edit2 / total) * 100).toFixed(2),
    correct,
    edit1,
    edit2,
    vocabMismatch,
    elapsedSeconds: parseFloat(elapsed),
    avgTimePerSample: parseFloat((parseFloat(elapsed) / total).toFixed(2)),
  };

  console.log("\n=== BENCHMARK RESULTS ===");
  console.log(`Dataset: ${results.dataset} (${results.totalSamples} samples)`);
  console.log(`Beam Width: ${results.beamWidth}`);
  console.log(
    `ExpRate (exact match): ${results.expRate}% (${results.correct}/${results.totalSamples})`,
  );
  console.log(`ExpRate ≤1 edit: ${results.expRate1}%`);
  console.log(`ExpRate ≤2 edits: ${results.expRate2}%`);
  console.log(`Vocab mismatch samples: ${results.vocabMismatch}`);
  console.log(
    `Total time: ${results.elapsedSeconds}s (${results.avgTimePerSample}s/sample)`,
  );

  // Save results
  const resultsDir = resolve(import.meta.dirname!, "results");
  writeFileSync(
    resolve(resultsDir, "benchmark.json"),
    JSON.stringify(results, null, 2),
  );
  writeFileSync(
    resolve(resultsDir, "errors.json"),
    JSON.stringify(errors.slice(0, 100), null, 2),
  );
  console.log("\nResults saved to benchmark/results/");

  // Cleanup
  await encSession.release();
  await decSession.release();
}

main().catch(console.error);
