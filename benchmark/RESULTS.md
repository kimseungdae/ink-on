# ink-on Benchmark Results

## CROHME 2014 Test Set

**Dataset**: 986 handwritten math expression samples from the CROHME 2014 competition
**Evaluation**: End-to-end (InkML strokes → preprocessing → ONNX inference → LaTeX comparison)
**Platform**: Node.js + onnxruntime-node + node-canvas (CPU)

### Results

| Model                     | ExpRate    | ≤1 edit    | ≤2 edits   | Size       | Runtime      |
| ------------------------- | ---------- | ---------- | ---------- | ---------- | ------------ |
| CoMER paper (FP32)        | 59.33%     | —          | —          | ~100 MB    | PyTorch/GPU  |
| **ink-on (INT8, beam=3)** | **36.41%** | **53.25%** | **65.82%** | **7.2 MB** | **ONNX/CPU** |

- **Exact match**: 359/986 (36.41%)
- **Within 1 token edit**: 525/986 (53.25%)
- **Within 2 token edits**: 649/986 (65.82%)
- **Vocab mismatch**: 125/986 (12.7%) — GT contains symbols outside ink-on's 113-token vocabulary
- **Average time per sample**: 0.29s (CPU, Node.js)

### Methodology Notes

1. **End-to-end evaluation**: Unlike the CoMER paper which evaluates on pre-rendered images, our benchmark runs the full ink-on pipeline: InkML stroke coordinates → canvas rendering → preprocessing → ONNX encoder → ONNX decoder → LaTeX output. This includes preprocessing differences that may affect results.

2. **LaTeX normalization**: Both ground truth and predictions are normalized to use explicit braces for subscripts/superscripts (`x_k` → `x_{k}`), remove LaTeX spacing commands (`\!`, `\,`), and expand single-argument `\frac`/`\sqrt` to braced form.

3. **Token-level edit distance**: ExpRate uses token-level (not character-level) comparison after normalization.

4. **Vocab coverage**: 125 samples (12.7%) contain symbols not in the 113-token vocabulary (e.g., `\!`, `D`, uppercase letters not included). These samples cannot achieve exact match.

### Factors Affecting ExpRate

- **INT8 quantization**: ~100 MB FP32 → 7.2 MB INT8 (92.8% reduction). Expected accuracy loss from quantization alone is typically 3-8%.
- **Preprocessing differences**: ink-on uses node-canvas for rendering, which may produce slightly different anti-aliasing compared to the Python/PIL preprocessing used during CoMER training.
- **Evaluation methodology**: End-to-end from strokes vs. paper's image-based evaluation.
- **Limited vocabulary**: 113 tokens vs. potentially larger training vocabulary.

### Reproduction

```bash
cd benchmark
npm install
npx tsx download-crohme.ts  # Download CROHME dataset
npx tsx run.ts              # Run benchmark (~5 minutes on modern CPU)
```

Results are saved to `benchmark/results/benchmark.json`.
