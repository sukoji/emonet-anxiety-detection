# Resume / Interview Summary

## Voice Phishing Prevention ATM Emotion Recognition System

**Team Tripos (트라이포스)** · Sangmyung University · Graduation Project · 2024

**Members:** Kim Seong-hyeon, Byeon Seong-ho, Ahn Seong-chan, Lim Jae-young, Jin Seok-ho

### One-Line

Developed an ATM emotion recognition system that detects anxiety from facial expressions in real time to help prevent voice phishing, using Knowledge Distillation and LDA+SVM.

### Problem

ATM voice phishing warnings are passive and often ineffective. We built an active detection pipeline: camera → emotion analysis → anxiety classification → counselor alert.

### Technical Highlights

- Lightweight FER via Knowledge Distillation (EmoNet teacher → MobileNetV2 student, 0.3 GMAC)
- Korean emotion dataset from AI-Hub (7 classes, ~500K samples)
- LDA + SVM anxiety classifier on arousal/valence (86% accuracy, 0.96 recall on anxiety)
- Tunable SVM sensitivity for emergency vs. monitoring modes
- Real-time multi-threaded webcam demo (`src/cap.py`)

### Outcome

Presented at 2024 Sangmyung University Graduation Portfolio Festival (Nov 2024).

### Reference

Lee, K.; Kim, S.; Lee, E.C. Fast and Accurate FER via Knowledge Distillation. *Appl. Sci.* 2023, 13, 6409.
