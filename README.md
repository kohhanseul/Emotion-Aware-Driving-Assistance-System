# 차두리 — 운전자 감정 기반 운전 도우미 (EffEmoteNet)

> 운전자의 얼굴 표정을 실시간으로 인식하고, 감정 상태에 따라 맞춤형 안전 조치를 제공하는 AI 운전 보조 시스템
>
> 🏆 ESTSOFT WASSUP AI 모델 개발자 과정 11기 · 비정형 데이터 경진대회 **대상 수상** (2025.12)

## 👥 팀 프로젝트(4인)

**본인 담당 역할**

- **모델링·실험**: ViT를 제외한 사전학습 CNN 모델 전반의 성능 비교 및 파인튜닝 전략 설계 — 파인튜닝 범위(top-k / % / 전체)와 학습률 조합 실험으로 최적 학습 전략 도출, EfficientNet 특징 추출 + ML 분류(LinearSVC) 하이브리드 실험
- **시스템 구현**: Streamlit 실시간 시연 UI, MetaDrive 시뮬레이터 초기 구현, LLM 단계 설계·프롬프트 구현(감정별 멘트·끝말잇기·경고 로직), 외부 API(Gemini, 우리말샘 사전) 연결 및 통합
- **아키텍처**: Vision 모델 → LLM → 차량/외부 알림으로 이어지는 전체 시스템 아키텍처 및 데이터 흐름 설계

※ 커스텀 모델(EffEmoteNet) 구조 설계는 팀 공동으로 진행했습니다.

---

## 문제 정의

분노·졸음·공황 같은 운전자 요인은 교통 사망사고의 주요 원인입니다. 기존 ADAS가 차량 움직임(차선 이탈, 급제동)에 의존하는 것과 달리, 본 프로젝트는 **운전자의 감정 상태를 직접 인식해 위험을 더 이른 단계에서 감지**하는 것을 목표로 했습니다.

- 운전자의 얼굴 표정에서 감정 상태(anger, closed, happy, panic, sadness)를 실시간 분류
- 정상 눈 깜빡임(0.1~0.4초)과 마이크로슬립(2초 이상)을 프레임 단위로 구분 (NHTSA 기준)
- 감정별 맞춤 대응 시나리오(갓길 정차 유도, 끝말잇기 각성 유도, 119 위치 전송 등) 실행

## 데이터셋

공개 얼굴 표정 데이터셋을 Roboflow에서 **수작업으로 재분류·정제**했습니다.

1. 눈 감은 이미지를 'closed' 클래스로 분리 (졸음 감지용 신규 클래스)
2. 오분류·애매한 이미지 이동/삭제
3. closed 데이터 15° 로테이션 3배 증강 (클래스 불균형 완화)
4. 1차 학습 후 성능 저하 원인이던 'pain' 클래스 제거 → **최종 5클래스**

| 클래스 | 운전 상황 해석 | train / valid / test |
|---|---|---|
| anger | 보복·난폭 운전 위험 | 1,353 / 266 / 233 |
| closed | 졸음·마이크로슬립 | 779 / 47 / 41 |
| happy | 안전 운전 상태 | 1,534 / 324 / 329 |
| panic | 돌발·응급 상황 | 1,437 / 257 / 269 |
| sadness | 주의력 저하 | 1,240 / 240 / 253 |

## 모델 선정 과정

총 12단계 실험으로 사전학습 모델 → 파인튜닝 전략 → 커스텀 모델 순서로 좁혀갔습니다.

**1) 사전학습 모델 전수 비교** — VGG16, DenseNet, InceptionV3, ResNet50, MobileNet, EfficientNetV2 6종을 동일 조건에서 비교 후, 상위 모델로 파인튜닝 범위(top20 레이어 / 30% / 100% / top250)와 학습률 조합 실험. EfficientNetV2 **top250 레이어 파인튜닝**(lr 0.00005)이 최고 성능(Acc 0.89).

**2) 특징 추출 + ML 하이브리드** — 파인튜닝된 EfficientNetV2의 dense층 특징을 LinearSVC로 분류, 소폭 추가 개선(Acc 0.889).

**3) 실시간성 검증과 커스텀 설계** — 정확도 최고였던 EfficientNetV2 계열이 추론 36ms+/frame로 30fps(33ms) 요건 미달. 이에 ResEmoteNet 구조의 Residual Block을 EfficientNet의 MBConvBlock으로 교체하고(φ=3 스케일링), Y-Sobel 엣지 채널과 CBAM을 결합한 **EffEmoteNet**을 설계 → 파라미터 약 2,400만 개 절감(80.2M → 56.0M).

### 📊 최종 결과 요약

| 모델 | Accuracy | F1 | 추론 속도 (frame당) | 비고 |
|---|---|---|---|---|
| EfficientNetV2 (top250 FT) + LinearSVC | 0.889 | 0.89 | 38.3ms | 정확도 최고, 실시간 미달 |
| EfficientNetV2-S (top250 FT) | 0.880 | 0.89 | 36.4ms | 실시간 미달 (~27fps) |
| ViT | 0.868 | 0.87 | 4.3ms | |
| **EffEmoteNet (+Y-Sobel+CBAM)** | **0.854** | **0.884** | **3.06ms** | **최종 선정** |

→ 정확도를 약 3%p 양보하는 대신 **추론 속도를 12배 단축**해, 30fps 환경에서 정상 깜빡임(3~12프레임)과 마이크로슬립(약 60프레임 연속 closed)을 모든 프레임 단위로 안정적으로 구분할 수 있는 모델을 선택했습니다.

🔗 전체 실험 결과(성능표 전체): https://docs.google.com/spreadsheets/d/1ZySsRmdCoauqzmTkgaegB4e8ZylHtR5F50vdyHLMNsM/edit?gid=1312453192#gid=1312453192

## EffEmoteNet 설계 포인트

- **MBConvBlock**: Inverted-Bottleneck 구조로 같은 깊이의 Residual Block 대비 파라미터 효율적 — 블록을 1.7배 더 쌓아 확장하면서도 전체 파라미터는 감소
- **Y-Sobel 엣지 채널**: 감은 눈의 'ㅡ' 형태 수평 엣지를 강조하기 위해 Grayscale + Sobel_Y 필터로 4번째 입력 채널 생성 (연산 최소화를 위해 Y 방향 단독 사용)
- **CBAM**: Channel Attention으로 엣지 정보가 담긴 채널을, Spatial Attention으로 얼굴 핵심 영역을 강조

실험 결과, 베이스라인(EfficientNet B3, ResEmoteNet)에서는 두 기법의 효과가 불안정했지만 **EffEmoteNet에서는 Sobel+CBAM 조합이 가장 일관된 성능 향상**을 보였습니다.

## 데모 구성

웹캠(블랙박스 대체) → 감정 분류(EffEmoteNet) → **LLM(Gemini 2.0 Flash-Lite)** 이 감정별 next_action 결정 → Streamlit 대시보드 + MetaDrive 차량 시뮬레이션

- closed: 외부 디스플레이 "졸음운전" 경고 + 끝말잇기 각성 유도 (끝말잇기 단어 검증에 우리말샘 사전 API 연동)
- panic: 응급 확인 멘트 → 무응답 시 119 위치 전송
- anger/sadness: 외부 "감정 주의 운전" 표시 + 명상/음악 안내
- happy: "안전운전" 표시

데모 영상(캠화면은 이모지로 대체)
- 메타드라이브(실시간 주행) : https://docs.google.com/videos/d/1RhgKeznt-Zrk0J7BEwpxz_rGhrtDS5pfoNsszPcpJG0/edit?scene=id.p#scene=id.p
- 끝말잇기 : https://drive.google.com/file/d/1MQOkhMHMt3ybiAn6L_pF3CEz1pHjPT2t/view?usp=sharing

## 모델 비교에서 얻은 인사이트

- **데이터 설계가 모델 성능을 결정한다** — pain 클래스 제거와 얼굴 크롭(MTCNN)이 모델 교체보다 더 큰 성능 개선을 가져옴
- **무거운 모델이 항상 좋은 선택은 아니다** — accuracy 1위 모델이 실시간 요건에서 탈락
- **Attention과 Edge 정보는 상호 보완적** — 단, 베이스 구조에 따라 효과가 갈림
- **LLM의 가치는 '대화'보다 '제어 로직'** — 분류 결과를 행동(경고·정차·안내)으로 연결할 때 시스템 완성도가 크게 향상

## 한계 및 향후 보완점

- 선글라스·마스크·역광, 야간/저조도 환경에 취약 → 실환경 데이터 보강, IR 카메라 등 보조 필요
- 감정 오인식 시 UX 저하 → 음성·생체신호 결합 멀티모달 확장 필요
- MBConvBlock 일부를 Fused-MBConv로 교체하는 EfficientNetV2 원리 적용 실험 예정

## 기술 스택

| 구분 | 사용 기술 |
|---|---|
| Language | Python |
| DL Framework | PyTorch (EffEmoteNet, ViT), TensorFlow (사전학습 모델 비교) |
| CV | OpenCV, MTCNN |
| LLM | Gemini 2.0 Flash-Lite (+ 우리말샘 사전 API) |
| Simulation | MetaDrive |
| UI | Streamlit |
| 실험 환경 | NVIDIA A40 (43GB), CUDA/cuDNN |

## 저장소 구성

```
├── demo_emotion.py            # 표정 인식 데모 (웹캠/이미지)
├── app/app.py                 # LLM 대응 시나리오 데모 (Streamlit)
├── models/
│   ├── effemotenet_infer.py   # 추론 전용 모델 정의
│   └── effemotenet_model.py   # 학습 당시 모델·실험 코드
├── scripts/export_inference_weights.py  # 학습 체크포인트 → 추론용 가중치 추출
├── preprocessing/face.py      # MTCNN 얼굴 크롭 전처리 (학습 데이터 제작용)
└── training/effemotenet_train.py
```

## 실행 방법

### 1) 표정 인식 데모 (EffEmoteNet)

학습된 가중치(`effemotenet_infer.pt`, 약 215MB / 56M 파라미터)는
**[GitHub Releases](../../releases)** 에서 다운로드해 `models/` 폴더에 넣어주세요.

```bash
pip install -r requirements.txt

python demo_emotion.py --webcam              # 웹캠 실시간 데모
python demo_emotion.py --image face.jpg      # 이미지 1장 분류
```

- `facenet-pytorch`가 설치되어 있으면 MTCNN으로 얼굴을 크롭해 입력하고, 없으면 프레임 전체를 사용합니다 (`pip install facenet-pytorch`, 선택).
- 표의 추론 속도 3.06ms는 NVIDIA A40 기준이며, CPU에서는 프레임당 수십 ms 수준입니다.

### 2) LLM 대응 시나리오 데모 (Streamlit)

감정 분류 결과에 따라 LLM이 경고·끝말잇기·119 연계 등 next_action을 결정하는 부분입니다.

```bash
cp .env.example .env   # GOOGLE_API_KEY, KOREAN_DICT_API_KEY 입력
streamlit run app/app.py
```

## 참고

본 저장소는 제안 시스템의 데모 구현입니다. 실시간 감정 인식, 모델 추론, UI 통합 등 핵심 구성 요소는 완전히 구현되었으며, 차량 제어 시나리오는 실현 가능성과 실시간 성능 검증을 목적으로 시뮬레이터 기반으로 부분 구현되었습니다.
