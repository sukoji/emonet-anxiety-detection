# Resume Project Summary

## Emotion-Based Anxiety Detection (Graduation Project)

실시간 카메라 입력에서 얼굴 표정을 분석하고, arousal/valence 기반으로 불안 상태를 분류하는 감정 AI 시스템을 구현했습니다.

### Key Contributions

- ONNX 기반 감정 추론 파이프라인을 실시간 웹캠 루프에 통합
- arousal/valence 특징을 활용한 LDA + SVM 불안 이진 분류기 설계
- 멀티스레드 + 큐 구조로 프레임 처리 지연 완화
- 사용자 이해도를 높이는 시각화 UI(상태, score, 바/좌표 오버레이) 구현

### Tech

Python, OpenCV, ONNX Runtime, scikit-learn, NumPy, Pandas

### Impact

- 단순 감정 라벨 출력에서 나아가 심리 상태 판단(불안 여부)으로 기능 확장
- 실시간 데모 가능한 형태로 구현해 프로젝트 완성도 향상
- 감정 컴퓨팅을 사용자 피드백 시스템으로 연결하는 실무형 구조 경험 확보

### One-Line

얼굴 표정 감정 인식 모델에 통계 기반 분류기를 결합해 실시간 불안 탐지 데모를 완성한 프로젝트.
