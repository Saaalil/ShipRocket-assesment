from __future__ import annotations

import numpy as np

from smart_turn.audio import prepare_audio


def maybe_augment(
    audio: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator | None = None,
    noise_prob: float = 0.3,
    gain_prob: float = 0.4,
) -> np.ndarray:
    """Lightweight, semantics-preserving augmentations for training."""
    waveform = prepare_audio(audio, sample_rate, normalize=False)
    rng = rng or np.random.default_rng()
    if rng.random() < gain_prob:
        gain = float(rng.uniform(0.7, 1.15))
        waveform = waveform * gain
    if rng.random() < noise_prob:
        snr_db = float(rng.uniform(10.0, 30.0))
        signal_power = np.mean(waveform**2) + 1e-8
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise = rng.normal(0.0, np.sqrt(noise_power), size=waveform.shape).astype(np.float32)
        waveform = waveform + noise
    return np.clip(waveform, -1.0, 1.0).astype(np.float32)
