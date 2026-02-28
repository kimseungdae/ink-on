# Show HN Post Draft

## Title

Show HN: ink-on – Free handwritten math recognition, 100% in-browser (7.2 MB ONNX)

## Body

I built a handwritten math recognition engine that runs entirely in the browser. No server, no API key, no data leaves the device.

It uses CoMER (ECCV 2022) quantized to INT8 ONNX models (7.2 MB total). IndexedDB caches them after first download.

- ExpRate: 36.41% on CROHME 2014 (vs 59.33% for the original FP32 model), 65.82% within 2 edits
- Framework-agnostic core: works with React, Vue, Svelte, vanilla JS
- Apache 2.0 licensed

Demo: https://ink-on.vercel.app
GitHub: https://github.com/kimseungdae/ink-on
npm: ink-on

Why: Existing options (MyScript, Mathpix) are commercial APIs that send student data to servers. For EdTech apps serving students, client-side recognition means automatic FERPA/GDPR/PIPA compliance — no data to protect because no data is collected.

Tech: DenseNet+Transformer encoder → autoregressive beam search decoder, WASM multi-threaded, dynamic input sizing reduces computation by up to 80%. Models are 7.2 MB (down from ~100 MB FP32, 92.8% reduction via INT8 quantization).

The accuracy-size trade-off is real — INT8 loses ~23% ExpRate vs FP32 — but 65.82% of predictions are within 2 token edits of correct, and for educational use where students write relatively clean expressions, it's practical.

Happy to answer questions about the ONNX conversion, browser WASM optimizations, or the CoMER architecture.

---

# dev.to 포스트 메타데이터

```yaml
title: "ink-on: Free, Private Handwriting Math Recognition in the Browser"
published: true
description: "A zero-cost, open-source alternative to MyScript for EdTech — runs entirely client-side with 7.2 MB ONNX models. No server, no API key, automatic FERPA/GDPR compliance."
tags: webdev, machinelearning, typescript, opensource
canonical_url: https://github.com/kimseungdae/ink-on
cover_image: # (demo screenshot)
```

---

# velog.io 제목

"ink-on: 서버 없이 브라우저에서 돌아가는 무료 수학 필기 인식 엔진"

---

# GeekNews (news.hada.io) 제목

"ink-on - 브라우저에서 100% 동작하는 수학 필기 인식 (7.2MB, 오픈소스)"
