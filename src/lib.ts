export { default as MathCanvas } from "./components/MathCanvas.vue";

export {
  preprocessStrokes,
  isStrokeMeaningful,
  InferenceEngine,
  loadVocab,
  decodeTokenIds,
  decodeToTokenArray,
  repairLatex,
  isCompleteExpression,
  fetchWithCache,
  getCachedModel,
  cacheModel,
} from "./core";

export type {
  Stroke,
  StrokePoint,
  PreprocessResult,
  RecognitionMode,
  RecognitionResult,
  InferenceEngineOptions,
  Vocab,
} from "./core";
