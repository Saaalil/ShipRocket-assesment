from __future__ import annotations

from typing import Any

import numpy as np

from smart_turn.constants import MAX_AUDIO_SAMPLES, SAMPLE_RATE


def decode_hf_audio(audio: Any) -> tuple[np.ndarray, int]:
    """Decode Hugging Face audio whether it is a dict or an AudioDecoder."""
    if isinstance(audio, dict):
        return to_mono(np.asarray(audio["array"], dtype=np.float32)), int(
            audio.get("sampling_rate", SAMPLE_RATE)
        )
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        data = samples.data
        if hasattr(data, "detach"):
            data = data.detach().cpu().numpy()
        rate = int(getattr(samples, "sample_rate", SAMPLE_RATE))
        return to_mono(np.asarray(data, dtype=np.float32)), rate
    try:
        array = np.asarray(audio["array"], dtype=np.float32)
        rate = int(audio["sampling_rate"])
        return to_mono(array), rate
    except Exception as exc:
        raise TypeError(f"Unsupported audio type: {type(audio)}") from exc


def to_mono(audio: np.ndarray) -> np.ndarray:
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim == 2:
        axis = 0 if waveform.shape[0] < waveform.shape[1] else 1
        return waveform.mean(axis=axis).astype(np.float32)
    raise ValueError(f"Unsupported audio shape: {waveform.shape}")


def resample(audio: np.ndarray, sample_rate: int, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    waveform = to_mono(audio)
    if sample_rate == target_rate:
        return waveform
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    duration = waveform.shape[0] / float(sample_rate)
    target_length = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, 1.0, num=waveform.shape[0], endpoint=False)
    target_x = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(target_x, source_x, waveform).astype(np.float32)


def peak_normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    waveform = to_mono(audio)
    max_abs = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if max_abs < 1e-8:
        return waveform
    return (waveform * (peak / max_abs)).astype(np.float32)


def take_last_seconds(audio: np.ndarray, max_samples: int = MAX_AUDIO_SAMPLES) -> np.ndarray:
    waveform = to_mono(audio)
    if waveform.shape[0] <= max_samples:
        return waveform
    return waveform[-max_samples:]


def left_pad(audio: np.ndarray, target_samples: int = MAX_AUDIO_SAMPLES) -> np.ndarray:
    waveform = to_mono(audio)
    if waveform.shape[0] >= target_samples:
        return waveform[-target_samples:]
    padding = np.zeros(target_samples - waveform.shape[0], dtype=np.float32)
    return np.concatenate([padding, waveform])


def prepare_audio(
    audio: np.ndarray,
    sample_rate: int,
    max_seconds: float = 8.0,
    normalize: bool = True,
) -> np.ndarray:
    waveform = resample(audio, sample_rate, SAMPLE_RATE)
    if normalize:
        waveform = peak_normalize(waveform)
    max_samples = int(max_seconds * SAMPLE_RATE)
    waveform = take_last_seconds(waveform, max_samples)
    return left_pad(waveform, max_samples)
