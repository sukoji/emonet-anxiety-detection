# Required Assets

`src/cap.py` 실행에 필요한 파일입니다. 용량·라이선스 때문에 Git에는 포함하지 않으며, 아래 경로에 직접 배치해야 합니다.

## 디렉터리 구조

```
assets/
├── models/
│   └── fer_student_kd.onnx              # KD 경량 FER 모델 (MobileNetV2 student)
├── data/
│   └── anxiety_classifier_labels.csv    # 불안 분류기 학습 라벨 (arousal, valence, emotion)
└── ui/
    └── valence_arousal_chart_bg.png     # Valence-Arousal 2D 차트 배경 (선택)
```

## 파일 설명

| 파일 | 설명 | 필수 |
|:---|:---|:---:|
| `models/fer_student_kd.onnx` | Lee et al. (2023) KD student model을 ONNX로 변환한 감정 추론 모델. 얼굴 이미지에서 arousal/valence를 출력합니다. | O |
| `data/anxiety_classifier_labels.csv` | LDA + SVM 불안 분류기 학습용 데이터. `arousal`, `valence`, `emotion` 컬럼 포함. | O |
| `ui/valence_arousal_chart_bg.png` | 실시간 데모 화면 좌하단 Valence-Arousal 좌표 그래프 배경 이미지. | X |

## Google Drive에서 가져올 때 (구 파일명)

Drive `emonet/emonet5/emonet5/` 폴더에 있는 기존 파일은 아래처럼 이름을 바꿔서 넣으면 됩니다.

| Drive 원본 파일명 | 이 저장소 경로 |
|:---|:---|
| `FER_student (1).onnx` | `assets/models/fer_student_kd.onnx` |
| `tag10.csv` | `assets/data/anxiety_classifier_labels.csv` |
| `image.png` | `assets/ui/valence_arousal_chart_bg.png` |
