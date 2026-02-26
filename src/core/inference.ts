import * as ort from "onnxruntime-web";
import type { PreprocessResult } from "./preprocessing";
import type { Vocab } from "./tokenizer";
import { decodeTokenIds } from "./tokenizer";

ort.env.wasm.numThreads = 1;
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
  executionProvider?: "webgpu" | "wasm";
}

const DEFAULT_MAX_STEPS = 50;
const REPEAT_LIMIT = 3;

function yieldToMain(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

export class InferenceEngine {
  private encoderSession: ort.InferenceSession | null = null;
  private decoderSession: ort.InferenceSession | null = null;
  private options: Required<InferenceEngineOptions>;
  private loading: Promise<void> | null = null;

  constructor(options: InferenceEngineOptions) {
    this.options = {
      maxDecodeSteps: DEFAULT_MAX_STEPS,
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
    const sessionOptions: ort.InferenceSession.SessionOptions = {
      executionProviders: [ep],
    };

    try {
      [this.encoderSession, this.decoderSession] = await Promise.all([
        ort.InferenceSession.create(this.options.encoderUrl, sessionOptions),
        ort.InferenceSession.create(this.options.decoderUrl, sessionOptions),
      ]);
    } catch {
      if (ep !== "wasm") {
        console.warn(
          `[InferenceEngine] ${ep} not available, falling back to wasm`,
        );
        const fallback: ort.InferenceSession.SessionOptions = {
          executionProviders: ["wasm"],
        };
        [this.encoderSession, this.decoderSession] = await Promise.all([
          ort.InferenceSession.create(this.options.encoderUrl, fallback),
          ort.InferenceSession.create(this.options.decoderUrl, fallback),
        ]);
      } else {
        throw new Error("Failed to create ONNX sessions");
      }
    }
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

    const encoderFeatures = encResult["encoder_features"];
    const encoderMask = encResult["encoder_mask"];
    const t1 = performance.now();

    const { sos, eos } = vocab.special_tokens;
    const tokenIds: number[] = [sos];
    let repeatCount = 0;
    let lastToken = -1;

    for (let step = 0; step < this.options.maxDecodeSteps; step++) {
      // Yield to main thread every 5 steps to prevent UI freeze
      if (step > 0 && step % 5 === 0) {
        await yieldToMain();
      }

      const inputIds = new ort.Tensor(
        "int64",
        BigInt64Array.from(tokenIds.map(BigInt)),
        [1, tokenIds.length],
      );

      const decResult = await this.decoderSession!.run({
        encoder_features: encoderFeatures,
        encoder_mask: encoderMask,
        input_ids: inputIds,
      });

      const logits = decResult["logits"].data as Float32Array;
      const vocabSize = vocab.vocab_size;
      const lastStepOffset = (tokenIds.length - 1) * vocabSize;

      let maxVal = -Infinity;
      let maxIdx = 0;
      for (let i = 0; i < vocabSize; i++) {
        const val = logits[lastStepOffset + i];
        if (val > maxVal) {
          maxVal = val;
          maxIdx = i;
        }
      }

      if (maxIdx === eos) break;

      // Stop on repeated tokens (sign of degenerate decoding)
      if (maxIdx === lastToken) {
        repeatCount++;
        if (repeatCount >= REPEAT_LIMIT) break;
      } else {
        repeatCount = 0;
      }
      lastToken = maxIdx;

      tokenIds.push(maxIdx);
    }

    const t2 = performance.now();
    const resultIds = tokenIds.slice(1);

    return {
      latex: decodeTokenIds(resultIds, vocab),
      tokenIds: resultIds,
      encoderMs: Math.round(t1 - t0),
      decoderMs: Math.round(t2 - t1),
      totalMs: Math.round(t2 - t0),
    };
  }

  dispose(): void {
    this.encoderSession?.release();
    this.decoderSession?.release();
    this.encoderSession = null;
    this.decoderSession = null;
    this.loading = null;
  }
}
