from __future__ import annotations

from functools import lru_cache

import numpy as np

from smart_turn.audio import prepare_audio
from smart_turn.constants import MAX_AUDIO_SECONDS, N_FRAMES, N_MELS, SAMPLE_RATE


@lru_cache(maxsize=1)
def _feature_extractor():
    from transformers import WhisperFeatureExtractor

    return WhisperFeatureExtractor(chunk_length=MAX_AUDIO_SECONDS)


def extract_log_mel(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    waveform = prepare_audio(audio, sample_rate)
    extractor = _feature_extractor()
    features = extractor(
        waveform,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np",
        padding="max_length",
        max_length=MAX_AUDIO_SECONDS * SAMPLE_RATE,
        truncation=True,
        do_normalize=True,
    )
    values = np.asarray(features["input_features"][0], dtype=np.float32)
    if values.shape != (N_MELS, N_FRAMES):
        raise ValueError(f"Unexpected feature shape {values.shape}, expected {(N_MELS, N_FRAMES)}")
    return values
