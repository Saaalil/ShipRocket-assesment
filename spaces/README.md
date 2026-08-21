---
title: Assesment SR
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.25.0
app_file: app.py
pinned: false
license: apache-2.0
suggested_hardware: zero-a10g
# Gradio SSR is disabled in app.py via GRADIO_SSR_MODE=0.
# Keep that env var false in Space settings if HF re-enables SSR.
---

# Shiprocket turn detection

Audio-native complete / incomplete classifier. Run it after VAD sees a pause.

Trained on official Smart Turn v3.2 English + Hindi/Marathi slices (84,223 clips).
Validation: accuracy 0.618, macro-F1 0.617, ROC-AUC 0.670.

- Code: https://github.com/Saaalil/ShipRocket-assesment
- Weights: https://huggingface.co/Saalil/Assesment_SR-model
