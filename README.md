# khuda_final — 냉장고 “싹쓸이” 감지(SAM3 기반 베이스라인)

냉장고 내부 영상을 입력으로 받아, 사람이 냉장고에서 **물체(음식)를 2개 이상 가져가는 행위**를 “싹쓸이(positive)”로 보고 감지하기 위한 프로젝트입니다.  
현재 구현은 **SAM3(Transformers) 기반 세그멘테이션/트래킹 + 규칙 기반 take_count 산출**(베이스라인)까지 구성되어 있습니다.

---

## 핵심 아이디어(현재 베이스라인)

- SAM3로 `objects in refrigerator` 프롬프트 기반 **인스턴스 마스크**를 프레임별로 얻습니다.
- 냉장고 내부 ROI(폴리곤)를 정의하고, 객체 마스크가 ROI **경계에 닿는 순간을 take 이벤트**로 간주합니다.
  - 안정장치: 경계 접촉이 `K` 프레임 연속이어야 take로 카운트(노이즈 완화)
  - 너무 작은 마스크는 무시(오탐 완화)
- 영상 단위 `take_count >= 2`이면 싹쓸이로 분류합니다.

ROI는 기본적으로 2560x1440 기준으로 하드코딩되어 있으며, 입력 해상도(예: 1280x720)에 맞춰 **자동 스케일링**됩니다.

---

## 요구사항

- Python(권장: 프로젝트의 `environment.yml` 기반 conda env)
- `torch`, `transformers`, `accelerate`
- 비디오 입출력: `opencv-python`(cv2)
- SAM3 체크포인트 접근 권한 및 다운로드(캐시 포함)
  - 클러스터 환경에서는 보통 `hf auth login` 또는 토큰 설정이 필요합니다.

> 참고: 이 프로젝트는 로컬 `sam3/` 패키지에 의존하지 않고, 기본적으로 `transformers`의 `facebook/sam3`를 사용합니다.

---

## 빠른 시작

### 1) 단일 영상 세그멘테이션 오버레이(확인용)

`data/예시영상.mp4`를 `objects in refrigerator`로 세그멘테이션하고 오버레이 mp4를 저장합니다.

```bash
python3 scripts/sam3_single_video_overlay.py
```

출력:
- `logs/sam3_single/예시영상_sam3_overlay.mp4`

---

## take_count 실행(배치 / 스트리밍)

### 배치(Pre-loaded) 실행

```bash
python3 run_take_counter.py
```

내부 설정은 `runners/take_counter_batch.py` 상단 “EDIT HERE” 블록에서 조정합니다.

### 스트리밍(프레임 단위) 실행

```bash
python3 run_take_counter_streaming.py
```

내부 설정은 `runners/take_counter_streaming.py`의 `run_streaming_demo()` 또는 `StreamingRunConfig`로 조정합니다.

---

## 스트리밍 평가(Precision/Recall/F1)

아래 두 폴더를 데이터셋으로 사용합니다.

- Negative(정상): `data/가져감(1개)_화각변환`  → 기대: `take_count <= 1`
- Positive(싹쓸이): `data/가져감(2개이상)_화각변환` → 기대: `take_count >= 2`

실행:

```bash
python3 run_take_counter_streaming_eval.py
```

결과 저장:
- `logs/streaming_eval_take_counter/eval_summary.json` (tp/fp/fn/tn, precision/recall/f1)
- `logs/streaming_eval_take_counter/eval_items.jsonl` (영상별 take_count/에러/출력 경로)
- 디버그 비디오/요약 JSON:
  - `logs/streaming_eval_take_counter/neg_1_or_0/debug/*.mp4`
  - `logs/streaming_eval_take_counter/pos_2_or_more/debug/*.mp4`
  - `logs/streaming_eval_take_counter/**/summaries/*.json`

---

## 주요 코드 위치

- Streaming 추론 엔진: `detection/sam3_transformers_streaming.py`
- 배치 추론(프레임별 결과 정규화 포함): `detection/sam3_transformers.py`
- 규칙 기반 take_count(경계 접촉 기반 + 안정장치): `rules/take_counter.py`
- ROI(폴리곤/스케일링): `utils/fridge_roi.py`
- 스트리밍 프레임 읽기(FPS 다운샘플/리사이즈): `utils/video_frame_stream.py`
- 디버그 비디오 writer(스트리밍): `utils/take_debug_video_streaming.py`
- 러너 모음: `runners/`
  - `runners/take_counter_batch.py`
  - `runners/take_counter_streaming.py`
  - `runners/take_counter_streaming_eval.py`
  - `runners/take_counter_runner.py` (모드 통합 호출)

---

## OOM(메모리) 관련 참고

SAM3는 트래킹 상태를 내부에 유지하므로, 영상 길이/검출 객체 수에 따라 VRAM이 증가할 수 있습니다.  
실험용 기본 설정은 OOM 완화를 위해 다음을 사용합니다.

- 입력 프레임 리사이즈(예: 1280x720)
- FPS 다운샘플(예: 15fps)
- ROI 밖 입력 마스킹(옵션)
- 프레임별 결과를 GPU에 쌓지 않도록 CPU로 정규화 저장

---

## 로그/출력

- 결과물은 기본적으로 `logs/` 아래에 저장됩니다.
- 디버그 비디오는 규칙 검증(오탐/누락 분석)에 필수이므로 항상 저장하도록 구성했습니다.

