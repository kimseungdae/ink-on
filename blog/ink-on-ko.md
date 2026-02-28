---
title: "ink-on: 서버 없이 브라우저에서 동작하는 무료 수학 필기 인식"
description: "EdTech를 위한 프라이버시 우선, 제로 비용 MyScript 대안 — 7.2MB ONNX 모델로 100% 클라이언트 사이드 실행"
tags: [webdev, machine-learning, edtech, privacy]
date: 2026-02-28
---

# ink-on: 서버 없이 브라우저에서 동작하는 무료 수학 필기 인식

## 문제: EdTech의 프라이버시 딜레마

교육 앱에서 손글씨 수학 수식 인식이 필요하다면, 선택지는 한정적이고 비쌉니다:

- **MyScript** — 업계 표준. 연 $500+ 라이선스, 모든 데이터가 서버로 전송.
- **Mathpix** — 건당 $0.01 API. 학생 30명 × 하루 20문제 = 하루 $6, 월 $120.
- **Google Cloud Vision** — API 비용 + 지연시간 + 학생 필기 데이터가 Google 서버에.

미성년자를 대상으로 하는 교육 앱에서 이런 구조는 컴플라이언스 문제를 만듭니다:

- **개인정보보호법 (한국)** — 개인정보 수집 시 명확한 고지와 동의 필요
- **FERPA (미국)** — 학생 교육 기록을 동의 없이 제3자에게 공개 금지
- **GDPR (EU)** — 미성년자 개인 데이터 처리 시 부모 동의 필수
- **COPPA (미국)** — 13세 미만 아동 데이터 수집 시 부모 동의 필수

학생의 손글씨를 제3자 서버에 보내는 모든 API 호출은 잠재적 컴플라이언스 위반입니다.

## 해결책: 아키텍처 수준의 프라이버시

**ink-on**은 **100% 브라우저에서 동작하는** 손글씨 수학 인식 엔진입니다. 서버 없음. API 키 없음. 데이터가 디바이스를 떠나지 않음.

- **총 7.2 MB** — INT8 양자화 모델, 첫 다운로드 후 IndexedDB에 캐시
- **Apache 2.0** — 상용 프로젝트에서도 영구 무료
- **프레임워크 독립** — React, Vue, Svelte, 바닐라 JS 모두 지원
- **오프라인 지원** — 모델 캐시 후 인터넷 불필요

**[라이브 데모 →](https://ink-on.vercel.app)**

인식이 전적으로 디바이스에서 일어나기 때문에, 보호할 데이터도 없고, 얻을 동의도 없고, 감사할 제3자도 없습니다. 프라이버시 준수가 절차적이 아닌 **구조적**이 됩니다.

## 기술 심층 분석

ink-on은 ECCV 2022에서 발표된 [CoMER](https://github.com/Green-Wood/CoMER) (Coverage-guided Multi-scale Encoder-decoder Transformer)를 ONNX Runtime Web으로 브라우저에서 실행합니다.

### 아키텍처

```
Stroke[] → 전처리 → 인코더 (DenseNet + Transformer)
                          ↓
                     디코더 (오토리그레시브 Transformer)
                          ↓
                     Token IDs → LaTeX 문자열
```

1. **전처리** — 사용자 스트로크를 CROHME 규격에 따라 오프스크린 캔버스에 렌더링 (검은 배경에 흰 스트로크). 높이 256px로 스케일링, 너비는 64px 배수로 동적 조정.

2. **인코더** — DenseNet 백본이 다중 스케일 특성을 추출하고, Transformer 인코더가 위치 임베딩과 함께 컨텍스트 특성 맵을 생성.

3. **디코더** — 커버리지 어텐션을 갖춘 오토리그레시브 Transformer가 LaTeX 토큰을 순차적으로 생성. 빔 서치로 최적 가설 선택.

### INT8 양자화: 92.8% 크기 감소

|                     | FP32 (원본) | INT8 (ink-on) |
| ------------------- | ----------- | ------------- |
| 인코더              | ~55 MB      | 3.4 MB        |
| 디코더              | ~45 MB      | 4.0 MB        |
| **합계**            | **~100 MB** | **7.2 MB**    |
| 브라우저 배포 가능? | 불가        | **가능**      |

### 브라우저 최적화

- **동적 입력 너비** — 텐서 너비가 콘텐츠에 맞게 조정. 간단한 "2"는 256×128 텐서로 처리되어 연산량 최대 80% 절감.
- **멀티스레드 WASM** — SharedArrayBuffer로 CPU 코어 간 병렬 실행 (2-4배 속도 향상).
- **IndexedDB 캐싱** — 모델을 한 번 다운로드 후 로컬 캐시. 재방문 시 밀리초 단위 로드.
- **논블로킹 디코딩** — 디코더가 몇 스텝마다 메인 스레드에 양보하여 UI 응답성 유지.

## 벤치마크: CROHME 2014

[CROHME 2014](https://www.isical.ac.in/~crohme/) 테스트셋 (986개 손글씨 수학 수식)에서 평가했습니다.

| 모델                      | ExpRate    | ≤1 edit    | ≤2 edits   | 크기       | 런타임        |
| ------------------------- | ---------- | ---------- | ---------- | ---------- | ------------- |
| CoMER 논문 (FP32)         | 59.33%     | —          | —          | ~100 MB    | PyTorch/GPU   |
| **ink-on (INT8, beam=3)** | **36.41%** | **53.25%** | **65.82%** | **7.2 MB** | **ONNX/WASM** |

> **INT8 정확도에 대한 참고**: FP32 대비 ~23% ExpRate 하락은 INT8 양자화(92.8% 크기 축소)와 평가 파이프라인 차이의 복합 효과입니다. ≤2 edit 정확도 65.82%는 대부분의 수식에서 거의 정확한 출력을 생성함을 보여줍니다. 986개 샘플 중 125개(12.7%)는 ink-on의 113개 토큰 어휘에 없는 기호를 포함하여 정확 일치가 불가능합니다.

## 컴플라이언스 분석

### 클라이언트 사이드가 중요한 이유

| 규제                | 요구사항                      | ink-on 준수 현황       |
| ------------------- | ----------------------------- | ---------------------- |
| 개인정보보호법 (KR) | 개인정보 수집 시 고지 및 동의 | ✓ 수집 자체가 없음     |
| FERPA (US)          | 학생 기록 제3자 공개 금지     | ✓ 데이터 전송 없음     |
| GDPR (EU)           | 데이터 최소화, 미성년자 동의  | ✓ 데이터 수집 없음     |
| COPPA (US)          | 13세 미만 부모 동의           | ✓ 운영자에게 전송 없음 |

서버 기반 인식에서는 모든 API 호출이 감사 대상입니다. ink-on에서는 감사할 것이 없습니다 — 손글씨 데이터는 사용자의 브라우저 이외 어디에도 존재하지 않습니다.

이것이 **Privacy by Architecture** 입니다: 컴플라이언스가 설정이나 정책이 아니라, 시스템의 구조적 속성입니다.

### 비용 비교

| 솔루션   | 월 비용 (학생 1,000명, 하루 50문제) | 데이터 외부 전송? |
| -------- | ----------------------------------- | ----------------- |
| MyScript | $500+ (라이선스)                    | 예                |
| Mathpix  | $15,000 ($0.01/건 × 5만건/일 × 30)  | 예                |
| ink-on   | **$0**                              | **아니오**        |

## 통합: 5분이면 충분

```bash
npm install ink-on onnxruntime-web
```

```typescript
import { InferenceEngine, preprocessStrokes, loadVocab } from "ink-on/core";

const vocab = await loadVocab("/models/comer/vocab.json");
const engine = new InferenceEngine({
  encoderUrl: "/models/comer/encoder_int8.onnx",
  decoderUrl: "/models/comer/decoder_int8.onnx",
  beamWidth: 3,
});
await engine.init();

const input = preprocessStrokes(strokes);
const { latex } = await engine.recognize(input, vocab);
// latex: "x ^ { 2 } + 1"
```

모델은 [GitHub Releases](https://github.com/kimseungdae/ink-on/releases)에서 다운로드하여 `public/models/comer/`에 배치하세요.

## 향후 계획

- **WebGPU 지원** — 호환 브라우저에서 2-5배 추가 속도 향상
- **더 높은 정확도 모델** — 더 큰 CoMER 변형 및 학습 데이터 증강 탐색
- **확장 어휘** — 화학, 물리 표기법으로 어휘 확장

## 링크

- **라이브 데모**: [ink-on.vercel.app](https://ink-on.vercel.app)
- **GitHub**: [github.com/kimseungdae/ink-on](https://github.com/kimseungdae/ink-on)
- **npm**: `npm install ink-on`
- **라이선스**: Apache 2.0
- **논문**: [CoMER (ECCV 2022)](https://github.com/Green-Wood/CoMER)

---

_ink-on은 오픈소스이며 기여를 환영합니다. 프라이버시 문제 없이 손글씨 인식이 필요한 EdTech 프로젝트에서 활용해 보세요._
