from __future__ import annotations

SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 8
MAX_AUDIO_SAMPLES = SAMPLE_RATE * MAX_AUDIO_SECONDS
N_MELS = 80
N_FRAMES = 800  # Whisper 10 ms frames over 8 seconds
WHISPER_ENCODER_POSITIONS = N_FRAMES // 2  # conv stride 2
DEFAULT_THRESHOLD = 0.5
FEATURE_VERSION = "whisper-tiny-logmel-v1"
GITHUB_REPO = "Saaalil/ShipRocket-assesment"
HF_MODEL_REPO = "Saalil/Assesment_SR-model"
HF_SPACE_REPO = "Saalil/Assesment_SR"
HF_ONNX_FILENAME = "model_fp32.onnx"
