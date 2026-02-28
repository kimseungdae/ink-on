import * as ort from "onnxruntime-web";

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;

const REPEAT_LIMIT = 3;

// Number mode: digits + basic operators + \frac + structural tokens
const NUMBER_MODE_ALLOWED: Set<number> = new Set([
  0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 50,
  53, 69, 78, 82, 83, 110, 112,
]);

interface InitMsg {
  type: "init";
  encoderUrl: string;
  decoderUrl: string;
  executionProvider: string;
}

interface RecognizeMsg {
  type: "recognize";
  id: number;
  tensor: Float32Array;
  mask: Uint8Array;
  height: number;
  width: number;
  maskHeight: number;
  maskWidth: number;
  vocabSize: number;
  sos: number;
  eos: number;
  beamWidth: number;
  maxSteps: number;
  mode: "auto" | "number" | "expression";
}

type WorkerMsg = InitMsg | RecognizeMsg;

let encoderSession: ort.InferenceSession | null = null;
let decoderSession: ort.InferenceSession | null = null;

async function fetchModel(url: string): Promise<ArrayBuffer> {
  const DB_NAME = "math-handwrite-models";
  const STORE = "onnx-models";

  try {
    const db = await new Promise<IDBDatabase>((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => req.result.createObjectStore(STORE);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    const cached = await new Promise<ArrayBuffer | undefined>(
      (resolve, reject) => {
        const tx = db.transaction(STORE, "readonly");
        const req = tx.objectStore(STORE).get(url);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      },
    );
    if (cached) return cached;

    const buf = await (await fetch(url)).arrayBuffer();
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(STORE, "readwrite");
      const req = tx.objectStore(STORE).put(buf, url);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error);
    });
    return buf;
  } catch {
    return (await fetch(url)).arrayBuffer();
  }
}

async function handleInit(msg: InitMsg) {
  try {
    const [encBuf, decBuf] = await Promise.all([
      fetchModel(msg.encoderUrl),
      fetchModel(msg.decoderUrl),
    ]);
    const opts: ort.InferenceSession.SessionOptions = {
      executionProviders: [msg.executionProvider],
    };
    try {
      [encoderSession, decoderSession] = await Promise.all([
        ort.InferenceSession.create(encBuf, opts),
        ort.InferenceSession.create(decBuf, opts),
      ]);
    } catch {
      const fallback: ort.InferenceSession.SessionOptions = {
        executionProviders: ["wasm"],
      };
      [encoderSession, decoderSession] = await Promise.all([
        ort.InferenceSession.create(encBuf, fallback),
        ort.InferenceSession.create(decBuf, fallback),
      ]);
    }
    self.postMessage({ type: "init-done" });
  } catch (err) {
    self.postMessage({
      type: "init-error",
      error: String(err),
    });
  }
}

function logSoftmax(
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

function applyModeMask(
  logProbs: Float64Array,
  vocabSize: number,
  allowed: Set<number> | null,
): void {
  if (!allowed) return;
  for (let i = 0; i < vocabSize; i++) {
    if (!allowed.has(i)) {
      logProbs[i] = -Infinity;
    }
  }
}

function topK(arr: Float64Array, k: number): number[] {
  const indices = Array.from({ length: arr.length }, (_, i) => i);
  indices.sort((a, b) => (arr[b] ?? 0) - (arr[a] ?? 0));
  return indices.slice(0, k);
}

async function runDecoder(
  ids: number[],
  encFeatures: ort.Tensor,
  encMask: ort.Tensor,
): Promise<Float32Array> {
  const inputIds = new ort.Tensor(
    "int64",
    BigInt64Array.from(ids.map(BigInt)),
    [1, ids.length],
  );
  const res = await decoderSession!.run({
    encoder_features: encFeatures,
    encoder_mask: encMask,
    input_ids: inputIds,
  });
  inputIds.dispose();
  return res["logits"]!.data as Float32Array;
}

interface Beam {
  logProb: number;
  ids: number[];
  finished: boolean;
}

interface BeamCandidate {
  ids: number[];
  logProb: number;
}

async function handleRecognize(msg: RecognizeMsg) {
  try {
    const t0 = performance.now();

    const pixelValues = new ort.Tensor("float32", msg.tensor, [
      1,
      1,
      msg.height,
      msg.width,
    ]);
    const pixelMask = new ort.Tensor("bool", msg.mask, [
      1,
      msg.maskHeight,
      msg.maskWidth,
    ]);

    const encResult = await encoderSession!.run({
      pixel_values: pixelValues,
      pixel_mask: pixelMask,
    });
    pixelValues.dispose();
    pixelMask.dispose();

    const encFeatures = encResult["encoder_features"]!;
    const encMask = encResult["encoder_mask"]!;
    const t1 = performance.now();

    const { sos, eos, vocabSize, beamWidth, maxSteps, mode } = msg;
    const allowedTokens = mode === "number" ? NUMBER_MODE_ALLOWED : null;
    let candidates: BeamCandidate[];

    if (beamWidth <= 1) {
      const ids: number[] = [sos];
      let repeatCount = 0;
      let lastToken = -1;
      for (let step = 0; step < maxSteps; step++) {
        const logits = await runDecoder(ids, encFeatures, encMask);
        const offset = (ids.length - 1) * vocabSize;
        const logProbs = logSoftmax(logits, offset, vocabSize);
        applyModeMask(logProbs, vocabSize, allowedTokens);

        let maxVal = -Infinity;
        let maxIdx = 0;
        for (let i = 0; i < vocabSize; i++) {
          if (logProbs[i]! > maxVal) {
            maxVal = logProbs[i]!;
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
        ids.push(maxIdx);
      }
      candidates = [{ ids: ids.slice(1), logProb: 0 }];
    } else {
      let beams: Beam[] = [{ logProb: 0, ids: [sos], finished: false }];
      for (let step = 0; step < maxSteps; step++) {
        const allCandidates: Beam[] = [];
        for (const beam of beams) {
          if (beam.finished) {
            allCandidates.push(beam);
            continue;
          }
          const logits = await runDecoder(beam.ids, encFeatures, encMask);
          const offset = (beam.ids.length - 1) * vocabSize;
          const logProbs = logSoftmax(logits, offset, vocabSize);
          applyModeMask(logProbs, vocabSize, allowedTokens);
          const topIndices = topK(logProbs, beamWidth * 2);

          for (const idx of topIndices) {
            const newLogProb = beam.logProb + (logProbs[idx] ?? 0);
            if (idx === eos) {
              allCandidates.push({
                logProb: newLogProb,
                ids: beam.ids,
                finished: true,
              });
            } else {
              const ids = beam.ids;
              let rCount = 0;
              for (let j = ids.length - 1; j >= 1; j--) {
                if (ids[j] === idx) rCount++;
                else break;
              }
              if (rCount >= REPEAT_LIMIT) {
                allCandidates.push({
                  logProb: newLogProb,
                  ids: beam.ids,
                  finished: true,
                });
              } else {
                allCandidates.push({
                  logProb: newLogProb,
                  ids: [...beam.ids, idx],
                  finished: false,
                });
              }
            }
          }
        }
        allCandidates.sort(
          (a, b) =>
            b.logProb / Math.max(b.ids.length, 1) -
            a.logProb / Math.max(a.ids.length, 1),
        );
        beams = allCandidates.slice(0, beamWidth);
        if (beams.every((b) => b.finished)) break;
      }
      const completed = beams.filter((b) => b.finished);
      const sorted = (completed.length > 0 ? completed : beams).sort(
        (a, b) =>
          b.logProb / Math.max(b.ids.length, 1) -
          a.logProb / Math.max(a.ids.length, 1),
      );
      candidates = sorted.slice(0, beamWidth).map((b) => ({
        ids: b.ids.slice(1),
        logProb: b.logProb,
      }));
    }

    encFeatures.dispose();
    encMask.dispose();

    const t2 = performance.now();

    self.postMessage({
      type: "recognize-done",
      id: msg.id,
      candidates,
      encoderMs: Math.round(t1 - t0),
      decoderMs: Math.round(t2 - t1),
      totalMs: Math.round(t2 - t0),
    });
  } catch (err) {
    self.postMessage({
      type: "recognize-error",
      id: msg.id,
      error: String(err),
    });
  }
}

self.onmessage = (e: MessageEvent<WorkerMsg>) => {
  const msg = e.data;
  if (msg.type === "init") handleInit(msg);
  else if (msg.type === "recognize") handleRecognize(msg);
};
