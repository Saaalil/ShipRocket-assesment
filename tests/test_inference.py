from __future__ import annotations

import numpy as np
import pytest

from smart_turn.evaluate import compute_metrics, selection_score
from smart_turn.inference import predict_turn


def test_metrics_on_perfect_scores() -> None:
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.9, 0.8]
    metrics = compute_metrics(y, p, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["false_complete_rate"] == 0.0
    assert selection_score(metrics) > 0.9


def test_predict_requires_or_falls_back(tmp_path, monkeypatch) -> None:
    audio = np.zeros(1600, dtype=np.float32)
    monkeypatch.setenv("SMART_TURN_ONNX_PATH", str(tmp_path / "missing.onnx"))
    with pytest.raises((FileNotFoundError, Exception)):
        predict_turn(audio, 16000, model_path=str(tmp_path / "missing.onnx"))
