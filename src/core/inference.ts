import * as ort from "onnxruntime-web";
import type { PreprocessResult } from "./preprocessing";
import type { Vocab } from "./tokenizer";
import { decodeTokenIds } from "./tokenizer";
import { fetchWithCache } from "./model-cache";

ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
ort.env.wasm.simd = true;

export interface RecognitionResult {
  latex: string;
  tokenIds: number[];
  encoderMs: number;
  decoderMs: number;
  totalMs: number;
}

export interface InferenceEngineOptions {
  encoderUrl: string;
  decoderUrl: string;
  maxDecodeSteps?: number;
  beamWidth?: number;
  executionProvider?: "webgpu" | "wasm";
}

const DEFAULT_MAX_STEPS = 50;
const REPEAT_LIMIT = 3;

function yieldToMain(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

interface Beam {
  logProb: number;
  ids: number[];
  finished: boolean;
}

export class InferenceEngine {
  private encoderSession: ort.InferenceSession | null = null;
  private decoderSession: ort.InferenceSession | null = null;
  private options: Required<InferenceEngineOptions>;
  private loading: Promise<void> | null = null;

  constructor(options: InferenceEngineOptions) {
    this.options = {
      maxDecodeSteps: DEFAULT_MAX_STEPS,
      beamWidth: 3,
      executionProvider: "wasm",
      ...options,
    };
  }

  async init(): Promise<void> {
    if (this.encoderSession && this.decoderSession) return;
    if (this.loading) return this.loading;
    this.loading = this._loadSessions();
    await this.loading;
  }

  private async _loadSessions(): Promise<void> {
    const ep = this.options.executionProvider;
    const [encBuf, decBuf] = await Promise.all([
      fetchWithCache(this.options.encoderUrl),
      fetchWithCache(this.options.decoderUrl),
    ]);
    const opts: ort.InferenceSession.SessionOptions = {
      executionProviders: [ep],
    };
    try {
      [this.encoderSession, this.decoderSession] = await Promise.all([
        ort.InferenceSession.create(encBuf, opts),
        ort.InferenceSession.create(decBuf, opts),
      ]);
    } catch {
      if (ep !== "wasm") {
        const fallback: ort.InferenceSession.SessionOptions = {
          executionProviders: ["wasm"],
        };
        [this.encoderSession, this.decoderSession] = await Promise.all([
          ort.InferenceSession.create(encBuf, fallback),
          ort.InferenceSession.create(decBuf, fallback),
        ]);
      } else {
        throw new Error("Failed to create ONNX sessions");
      }
    }
  }

  private async runDecoder(
    encoderFeatures: ort.Tensor,
    encoderMask: ort.Tensor,
    ids: number[],
  ): Promise<Float32Array> {
    const inputIds = new ort.Tensor(
      "int64",
      BigInt64Array.from(ids.map(BigInt)),
      [1, ids.length],
    );
    const res = await this.decoderSession!.run({
      encoder_features: encoderFeatures,
      encoder_mask: encoderMask,
      input_ids: inputIds,
    });
    return res["logits"]!.data as Float32Array;
  }

  private logSoftmax(
    logits: Float32Array,
    offset: number,
    size: number,
  ): Float64Array {
    const result = new Float64Array(size);
    let max = -Infinity;
    for (let i = 0; i < size; i++) {
      const v = logits[offset + i]!;
      if (v > max) max = v;
    }
    let sumExp = 0;
    for (let i = 0; i < size; i++) {
      sumExp += Math.exp(logits[offset + i]! - max);
    }
    const logSumExp = Math.log(sumExp);
    for (let i = 0; i < size; i++) {
      result[i] = logits[offset + i]! - max - logSumExp;
    }
    return result;
  }

  private topK(arr: Float64Array, k: number): number[] {
    const indices = Array.from({ length: arr.length }, (_, i) => i);
    indices.sort((a, b) => (arr[b] ?? 0) - (arr[a] ?? 0));
    return indices.slice(0, k);
  }

  async recognize(
    input: PreprocessResult,
    vocab: Vocab,
  ): Promise<RecognitionResult> {
    await this.init();
    const t0 = performance.now();

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

    const encResult = await this.encoderSession!.run({
      pixel_values: pixelValues,
      pixel_mask: pixelMask,
    });
    const encoderFeatures = encResult["encoder_features"]!;
    const encoderMask = encResult["encoder_mask"]!;
    const t1 = performance.now();

    const { sos, eos } = vocab.special_tokens;
    const vocabSize = vocab.vocab_size;
    const beamWidth = this.options.beamWidth;
    let resultIds: number[];

    if (beamWidth <= 1) {
      resultIds = await this.greedyDecode(
        encoderFeatures,
        encoderMask,
        sos,
        eos,
        vocabSize,
      );
    } else {
      resultIds = await this.beamDecode(
        encoderFeatures,
        encoderMask,
        sos,
        eos,
        vocabSize,
        beamWidth,
      );
    }

    const t2 = performance.now();

    return {
      latex: decodeTokenIds(resultIds, vocab),
      tokenIds: resultIds,
      encoderMs: Math.round(t1 - t0),
      decoderMs: Math.round(t2 - t1),
      totalMs: Math.round(t2 - t0),
    };
  }

  private async greedyDecode(
    encoderFeatures: ort.Tensor,
    encoderMask: ort.Tensor,
    sos: number,
    eos: number,
    vocabSize: number,
  ): Promise<number[]> {
    const tokenIds: number[] = [sos];
    let repeatCount = 0;
    let lastToken = -1;

    for (let step = 0; step < this.options.maxDecodeSteps; step++) {
      if (step > 0 && step % 5 === 0) await yieldToMain();

      const logits = await this.runDecoder(
        encoderFeatures,
        encoderMask,
        tokenIds,
      );
      const offset = (tokenIds.length - 1) * vocabSize;
      let maxVal = -Infinity;
      let maxIdx = 0;
      for (let i = 0; i < vocabSize; i++) {
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

  private async beamDecode(
    encoderFeatures: ort.Tensor,
    encoderMask: ort.Tensor,
    sos: number,
    eos: number,
    vocabSize: number,
    beamWidth: number,
  ): Promise<number[]> {
    let beams: Beam[] = [{ logProb: 0, ids: [sos], finished: false }];

    for (let step = 0; step < this.options.maxDecodeSteps; step++) {
      if (step > 0 && step % 3 === 0) await yieldToMain();

      const candidates: Beam[] = [];
      for (const beam of beams) {
        if (beam.finished) {
          candidates.push(beam);
          continue;
        }
        const logits = await this.runDecoder(
          encoderFeatures,
          encoderMask,
          beam.ids,
        );
        const offset = (beam.ids.length - 1) * vocabSize;
        const logProbs = this.logSoftmax(logits, offset, vocabSize);
        const topIndices = this.topK(logProbs, beamWidth * 2);

        for (const idx of topIndices) {
          const newLogProb = beam.logProb + (logProbs[idx] ?? 0);
          if (idx === eos) {
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

  dispose(): void {
    this.encoderSession?.release();
    this.decoderSession?.release();
    this.encoderSession = null;
    this.decoderSession = null;
    this.loading = null;
  }
}
