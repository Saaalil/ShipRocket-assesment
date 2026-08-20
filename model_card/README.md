---
language: multilingual
license: apache-2.0
library_name: onnx
pipeline_tag: audio-classification
tags:
  - turn-detection
  - voice-ai
  - whisper
---

# Shiprocket Turn Detection

Audio-native complete/incomplete classifier built for the Shiprocket assignment.

## Intended use

Run after VAD detects a pause. Input: up to 8 seconds of 16 kHz mono audio. Output: `P(complete)`.

## Training data

Only `pipecat-ai/smart-turn-data-v3.2-train`. Indic results use official Hindi/Marathi slices.

## Limitations

- Not a full-duplex conversational model.
- Clip metrics do not prove live interruption quality.
- Browser preprocessing is approximate; use the Python path as reference.
- Do not use for safety-critical barge-in.

## Metrics

Fill after the first trained export. Until then the demo may fall back to Pipecat Smart Turn v3.2.
