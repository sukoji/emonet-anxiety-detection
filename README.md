# Emotion-Based Anxiety Detection

졸업 프로젝트: 실시간 얼굴 표정 분석으로 arousal/valence를 추정하고, LDA + SVM으로 불안(Anxiety) 상태를 분류하는 시스템입니다.

## Highlights

- 웹캠 실시간 추론 파이프라인 (멀티스레드 + 큐)
- ONNX 기반 감정 모델 추론
- arousal/valence -> 불안 이진 분류 + score 산출
- 실시간 시각화 UI (상태 텍스트, 박스 색상, 그래프 오버레이)

## Tech Stack

`Python` `OpenCV` `ONNX Runtime` `scikit-learn` `NumPy` `Pandas`

## Repository Structure

```text
.
├── src/cap.py              # 최종 실시간 데모 (emonet/emonet5/emonet5 기준)
├── assets/                 # 로컬 모델/데이터 (Git 미포함, README 참고)
├── docs/                   # 포트폴리오 상세 문서
├── docs/web/index.html     # 웹 포트폴리오 페이지
└── requirements.txt
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

1. `assets/README.md`를 보고 필요한 파일을 `assets/`에 배치
2. 실행:

```bash
python src/cap.py
```

3. 종료: `q` 키

## Final Version Criteria

Drive `emonet` 폴더 내 여러 버전 중 아래를 최종본으로 채택했습니다.

- 경로: `emonet/emonet5/emonet5/`
- 근거: 최신 수정일(2024-10), ONNX 통합, 실시간 불안 분류/시각화 완성

## Documentation

- [상세 포트폴리오](docs/PORTFOLIO.md)
- [이력서용 요약](docs/RESUME_SUMMARY.md)
- [웹 포트폴리오](docs/web/index.html)

## License Note

EmoNet 기반 코드는 원 프로젝트 라이선스(CC BY-NC-ND)를 따릅니다.  
본 저장소는 졸업 프로젝트 포트폴리오 목적의 통합/응용 구현을 정리한 형태입니다.
