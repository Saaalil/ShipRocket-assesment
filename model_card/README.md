---
language: multilingual
license: apache-2.0
library_name: onnx
pipeline_tag: audio-classification
tags:
  - turn-detection
  - whisper
  - onnx
---

# Shiprocket Turn Detection

Whisper Tiny encoder + attention pooling + MLP head.
Trained on official Smart Turn v3.2 English + Hindi/Marathi (84,223 clips).

Validation (n=8422, threshold 0.5): accuracy 0.618, macro-F1 0.617, ROC-AUC 0.670.

Serve `model_fp32.onnx`. See `projectspec.md` in the code repository.
