export { default as MathCanvas } from "./components/MathCanvas.vue";

export {
  preprocessStrokes,
  isStrokeMeaningful,
  InferenceEngine,
  loadVocab,
  decodeTokenIds,
  fetchWithCache,
  getCachedModel,
  cacheModel,
} from "./core";

export type {
  Stroke,
  StrokePoint,
  PreprocessResult,
  RecognitionResult,
  InferenceEngineOptions,
  Vocab,
} from "./core";
