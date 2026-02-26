import type { PreprocessResult } from "./preprocessing";
import type { Vocab } from "./tokenizer";
import { decodeTokenIds } from "./tokenizer";

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

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
}

export class InferenceEngine {
  private worker: Worker | null = null;
  private options: Required<InferenceEngineOptions>;
  private initPromise: Promise<void> | null = null;
  private nextId = 0;
  private pending = new Map<number, PendingRequest>();

  constructor(options: InferenceEngineOptions) {
    this.options = {
      maxDecodeSteps: DEFAULT_MAX_STEPS,
      beamWidth: 3,
      executionProvider: "wasm",
      ...options,
    };
  }

  async init(): Promise<void> {
    if (this.worker) return;
    if (this.initPromise) return this.initPromise;
    this.initPromise = this._init();
    await this.initPromise;
  }

  private async _init(): Promise<void> {
    const worker = new Worker(
      new URL("./inference.worker.ts", import.meta.url),
      { type: "module" },
    );

    await new Promise<void>((resolve, reject) => {
      const onMsg = (e: MessageEvent) => {
        if (e.data.type === "init-done") {
          cleanup();
          resolve();
        } else if (e.data.type === "init-error") {
          cleanup();
          reject(new Error(e.data.error));
        }
      };
      const onErr = (err: ErrorEvent) => {
        cleanup();
        reject(new Error(err.message));
      };
      const cleanup = () => {
        worker.removeEventListener("message", onMsg);
        worker.removeEventListener("error", onErr);
      };
      worker.addEventListener("message", onMsg);
      worker.addEventListener("error", onErr);
      worker.postMessage({
        type: "init",
        encoderUrl: this.options.encoderUrl,
        decoderUrl: this.options.decoderUrl,
        executionProvider: this.options.executionProvider,
      });
    });

    worker.onmessage = (e: MessageEvent) => {
      const data = e.data;
      if (data.type === "recognize-done" || data.type === "recognize-error") {
        const p = this.pending.get(data.id);
        if (!p) return;
        this.pending.delete(data.id);
        if (data.type === "recognize-error") {
          p.reject(new Error(data.error));
        } else {
          p.resolve(data);
        }
      }
    };

    this.worker = worker;
  }

  async recognize(
    input: PreprocessResult,
    vocab: Vocab,
  ): Promise<RecognitionResult> {
    await this.init();

    const id = this.nextId++;
    const { sos, eos } = vocab.special_tokens;
    const tensorCopy = new Float32Array(input.tensor);
    const maskCopy = new Uint8Array(input.mask);

    const result = await new Promise<{
      tokenIds: number[];
      encoderMs: number;
      decoderMs: number;
      totalMs: number;
    }>((resolve, reject) => {
      this.pending.set(id, {
        resolve: resolve as (v: unknown) => void,
        reject,
      });
      this.worker!.postMessage(
        {
          type: "recognize",
          id,
          tensor: tensorCopy,
          mask: maskCopy,
          height: input.height,
          width: input.width,
          vocabSize: vocab.vocab_size,
          sos,
          eos,
          beamWidth: this.options.beamWidth,
          maxSteps: this.options.maxDecodeSteps,
        },
        [tensorCopy.buffer, maskCopy.buffer],
      );
    });

    return {
      latex: decodeTokenIds(result.tokenIds, vocab),
      tokenIds: result.tokenIds,
      encoderMs: result.encoderMs,
      decoderMs: result.decoderMs,
      totalMs: result.totalMs,
    };
  }

  dispose(): void {
    this.worker?.terminate();
    this.worker = null;
    this.initPromise = null;
    for (const [, p] of this.pending) {
      p.reject(new Error("Engine disposed"));
    }
    this.pending.clear();
  }
}
