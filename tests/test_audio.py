from __future__ import annotations

import numpy as np

from smart_turn.audio import left_pad, prepare_audio, resample, take_last_seconds, to_mono
from smart_turn.constants import MAX_AUDIO_SAMPLES, SAMPLE_RATE


def test_to_mono_averages_channels() -> None:
    stereo = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    mono = to_mono(stereo)
    assert mono.shape == (2,)
    assert np.allclose(mono, 0.5)


def test_resample_changes_length() -> None:
    audio = np.ones(8000, dtype=np.float32)
    out = resample(audio, 8000, SAMPLE_RATE)
    assert out.shape[0] == SAMPLE_RATE


def test_take_last_seconds_and_left_pad() -> None:
    audio = np.arange(MAX_AUDIO_SAMPLES + 100, dtype=np.float32)
    clipped = take_last_seconds(audio)
    assert clipped.shape[0] == MAX_AUDIO_SAMPLES
    padded = left_pad(np.ones(10, dtype=np.float32))
    assert padded.shape[0] == MAX_AUDIO_SAMPLES
    assert padded[0] == 0
    assert padded[-1] == 1


def test_prepare_audio_is_8s_16k() -> None:
    audio = np.random.default_rng(0).normal(0, 0.1, size=3000).astype(np.float32)
    prepared = prepare_audio(audio, 8000)
    assert prepared.shape == (MAX_AUDIO_SAMPLES,)
    assert prepared.dtype == np.float32
