<div align="center">

# Emotion-Based Anxiety Detection

**졸업 프로젝트** · 상명대학교 · 2024

실시간 얼굴 표정 분석으로 arousal/valence를 추정하고, 불안 상태를 분류하는 시스템

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.16+-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

[상세 포트폴리오](docs/PORTFOLIO.md) · [이력서 요약](docs/RESUME_SUMMARY.md) · [웹 포트폴리오](docs/web/index.html)

</div>

---

## 프로젝트 개요

상명대학교 인간중심인공지능학과 졸업 프로젝트로, 2024년 3월부터 10월까지 진행했습니다. Knowledge Distillation 기반 경량 FER 모델을 활용해 실시간 불안 상태를 탐지하는 것이 주제이며, 최종 구현은 `src/cap.py`입니다. 참고 논문은 [Lee et al., *Appl. Sci.* 2023, 13, 6409](https://doi.org/10.3390/app13116409)입니다.

본 프로젝트는 [Lee et al. (2023)](https://doi.org/10.3390/app13116409)에서 제안한 **Knowledge Distillation + Teacher Bound** 기반 경량 얼굴 표정 인식(FER) 모델을 응용한 졸업 프로젝트입니다.

논문에서는 EmoNet을 teacher model로, MobileNetV2/EfficientFormer를 student model로 사용해 AffectNet 데이터셋에서 8종 감정 분류와 valence/arousal 회귀를 동시에 학습했습니다. 본 프로젝트는 이 student model(`fer_student_kd.onnx`)이 출력하는 arousal/valence 값을 입력으로 받아, **LDA + SVM** 분류기로 불안 여부를 실시간 판별하는 응용 시스템을 구현했습니다.

---

## 배경

Lee et al. (2023)의 논문은 이산적 감정 분류만으로는 감정 상태를 충분히 표현할 수 없다는 점에서 출발합니다. valence(긍·부정성)와 arousal(각성도)을 연속값으로 함께 추정해야 실제 응용이 가능하다는 전제입니다.

논문의 핵심 기여는 다음과 같습니다.

- EmoNet(teacher, 16.99 GMAC)의 지식을 MobileNetV2 student(0.3 GMAC)로 전달
- Teacher Bound loss로 분류와 회귀를 동시에 학습
- EmoNet 대비 연산량 56.63배 감소, 정확도 하락 1.64%p 이내
- CPU 환경에서도 실시간 추론 가능

본 졸업 프로젝트는 이 경량 student model의 출력을 한 단계 더 활용해, arousal/valence 공간에서 **불안/안정 이진 분류**를 수행하고 웹캠 데모로 시각화합니다.

---

## 시스템 구조

### 전체 파이프라인

```mermaid
flowchart LR
    CAM["웹캠 입력"] --> FACE["Haar Cascade\n얼굴 검출"]
    FACE --> ONNX["fer_student_kd.onnx\n(KD Student Model)"]
    ONNX --> AV["arousal / valence"]
    AV --> LDA["LDA + SVM\n불안 분류"]
    LDA --> UI["실시간 시각화\nAnxiety Yes/No + Score"]
```

### 논문 모델과 본 프로젝트의 관계

```
[Lee et al. 2023 — Knowledge Distillation]
  EmoNet (Teacher, 16.99 GMAC)
       │ KD + Teacher Bound
       ▼
  fer_student_kd.onnx (MobileNetV2, 0.3 GMAC)
       │ ONNX 변환
       ▼
  fer_student_kd.onnx  ──→  arousal, valence 출력
                                    │
[본 졸업 프로젝트 — 응용 시스템]        │
  anxiety_classifier_labels.csv       │
       │                             │
       ▼                             ▼
  LDA + SVM 학습  ←──── arousal, valence
       │
       ▼
  Anxiety: Yes / No  +  Score (0–100)
       │
       ▼
  웹캠 실시간 데모 (cap.py)
```

### 멀티스레드 처리

실시간 지연을 줄이기 위해 프레임 수집과 추론을 분리했습니다.

| 스레드 | 역할 | 큐 |
|:---|:---|:---|
| Webcam Thread | 프레임 읽기, 좌우 반전, 화면 렌더링 | `frame_queue` (max 16) |
| Batch Thread | 얼굴 검출, ONNX 추론, LDA+SVM 분류 | `result_queue` (max 8) |

큐가 가득 차면 오래된 프레임을 버려 최신 상태를 유지합니다.

---

## 핵심 기술

### 1. 경량 FER 모델 (논문 기반)

Lee et al. (2023)에서 학습한 student model을 ONNX로 변환해 CPU에서 실시간 추론합니다.

| 항목 | EmoNet (Teacher) | FER_student (Student) |
|:---|:---:|:---:|
| Backbone | EmoNet | MobileNetV2 |
| MAC | 16.99 G | 0.3 G |
| Expression Accuracy | 75% | 73.36% |
| Valence RMSE | 0.29 | 0.37 |
| Arousal RMSE | 0.27 | 0.30 |
| CPU 실시간 | 불가 | 가능 |

출처: Lee et al., *Appl. Sci.* 2023, Table 3

### 2. 불안 분류기 (본 프로젝트 구현)

논문 모델이 출력한 arousal/valence를 불안 여부로 변환하는 2단계 분류기입니다.

```
arousal, valence
  → SimpleImputer (결측치 처리)
  → StandardScaler (표준화)
  → LDA (차원 축소)
  → SVM (RBF, 이진 분류)
  → Anxiety: Yes(1) / No(0) + Score (0–100)
```

**감정 레이블 매핑** (7종 → 2종)

| 원본 감정 | 코드 | 분류 |
|:---|:---:|:---:|
| 기쁨, 중립 | 0, 5 | 안정 (0) |
| 당황, 분노, 불안, 상처, 슬픔 | 1–4, 6 | 불안 (1) |

### 3. 시각화

| 요소 | 설명 |
|:---|:---|
| 얼굴 박스 | 안정: 녹색 / 불안: 빨간색 |
| 상태 텍스트 | `Anxiety: Yes/No` + Score |
| Valence 바 | 얼굴 좌측 세로 막대 |
| Arousal 바 | 얼굴 하단 가로 막대 |
| 2D 좌표 | Valence-Arousal 평면 위 점 |

---

## 기술 스택

| 영역 | 기술 | 용도 |
|:---|:---|:---|
| 언어 | Python 3.10+ | 전체 파이프라인 |
| 컴퓨터 비전 | OpenCV 4.8+ | 웹캠 I/O, Haar Cascade, UI |
| 딥러닝 | ONNX Runtime, PyTorch | FER student model 추론 |
| 머신러닝 | scikit-learn (LDA, SVM) | 불안 이진 분류 |
| 데이터 | NumPy, Pandas | 전처리, 라벨 관리 |
| 동시성 | threading, queue | 실시간 멀티스레드 파이프라인 |

---

## 프로젝트 구조

```
emonet-anxiety-detection/
├── README.md
├── requirements.txt
├── src/
│   └── cap.py                  # 최종 실시간 데모
├── assets/                     # 로컬 전용 (Git 미포함)
│   ├── models/
│   │   └── fer_student_kd.onnx           # KD 경량 FER 모델
│   ├── data/
│   │   └── anxiety_classifier_labels.csv # 불안 분류기 학습 라벨
│   └── ui/
│       └── valence_arousal_chart_bg.png  # 차트 배경 (선택)
└── docs/
    ├── PORTFOLIO.md
    ├── RESUME_SUMMARY.md
    └── web/index.html
```

---

## 실행 방법

```bash
git clone https://github.com/sukoji/emonet-anxiety-detection.git
cd emonet-anxiety-detection

python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# assets/ 에 모델·라벨 파일 배치 (assets/README.md 참고)
#   models/fer_student_kd.onnx
#   data/anxiety_classifier_labels.csv
python src/cap.py
```

종료: 화면 포커스 상태에서 `q` 키

---

## 개발 과정

```
2024.03  EmoNet 기반 기본 감정 인식 (video.py, test.py)
2024.04  한국어 감정 레이블링 (7종: 기쁨·당황·분노·불안·상처·슬픔·중립)
2024.05  ibug 파이프라인 통합 (RetinaFace + FAN + EmoNet)
2024.09  KD student model ONNX 변환 실험
2024.10  최종본 cap.py — ONNX CPU 추론 + LDA/SVM 불안 분류 + 실시간 UI
```

Drive `emonet` 폴더 내 8개 버전 중 `emonet/emonet5/emonet5/`를 최종본으로 선정했습니다. 선정 기준은 최신 수정일(2024-10-28), ONNX 통합 완료, 단일 파일 실행 가능 여부입니다.

---

## 담당 역할

| 영역 | 내용 |
|:---|:---|
| 모델 응용 | Lee et al. (2023) student model을 ONNX로 변환하고 실시간 파이프라인에 통합 |
| 불안 분류기 | arousal/valence 기반 LDA + SVM 이진 분류기 설계 |
| 실시간 최적화 | 멀티스레드 + 큐 구조로 프레임 지연 완화 |
| 시각화 | 불안 상태 색상 코딩, Valence-Arousal 2D 그래프 |
| 데이터 | 한국어 감정 레이블 7종 수집·정제·CSV 변환 |

---

## 한계 및 향후 과제

- 단일 모달리티(영상)만 사용. 음성·텍스트 결합 여지 있음
- AffectNet 기반 학습으로 일반화 검증이 부족
- 사용자별 baseline 없이 절대값으로 판단
- 임상 진단 목적으로 사용할 수 없음

향후에는 시계열 모델 도입, 사용자별 baseline 적응, 정량 평가(Accuracy, F1, Latency) 체계화를 검토할 수 있습니다.

---

## 참고 문헌

**주요 참고 논문 (졸업 프로젝트 기반 연구)**

```bibtex
@article{lee2023fast,
  author  = {Lee, Kunyoung and Kim, Seunghyun and Lee, Eui Chul},
  title   = {Fast and Accurate Facial Expression Image Classification
             and Regression Method Based on Knowledge Distillation},
  journal = {Applied Sciences},
  volume  = {13},
  number  = {11},
  pages   = {6409},
  year    = {2023},
  doi     = {10.3390/app13116409}
}
```

**Teacher Model**

```bibtex
@article{toisoul2021estimation,
  author  = {Toisoul, Antoine and Kossaifi, Jean and Bulat, Adrian
             and Tzimiropoulos, Georgios and Pantic, Maja},
  title   = {Estimation of continuous valence and arousal levels
             from faces in naturalistic conditions},
  journal = {Nature Machine Intelligence},
  year    = {2021},
  doi     = {10.1038/s42256-020-00280-0}
}
```

---

## 라이선스

EmoNet 기반 코드는 [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) 라이선스를 따릅니다.  
Lee et al. (2023) 논문은 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 오픈 액세스입니다.

---

<div align="center">

졸업 프로젝트 · 상명대학교 · 2024

</div>
