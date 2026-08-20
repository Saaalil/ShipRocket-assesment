from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import WhisperConfig

from smart_turn.export import export_onnx
from smart_turn.model import SmartTurnModel


def test_onnx_export_roundtrip(tmp_path) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    config = WhisperConfig(
        vocab_size=100,
        num_mel_bins=80,
        encoder_layers=1,
        encoder_attention_heads=4,
        decoder_layers=1,
        decoder_attention_heads=4,
        d_model=32,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_source_positions=10,
        num_hidden_layers=1,
    )
    model = SmartTurnModel(config).eval()
    onnx_path = tmp_path / "tiny.onnx"
    export_onnx(model, onnx_path)
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    features = np.zeros((1, 80, 20), dtype=np.float32)
    with torch.no_grad():
        torch_prob = model(torch.from_numpy(features))["probabilities"].numpy().reshape(-1)[0]
    onnx_prob = session.run(None, {session.get_inputs()[0].name: features})[0].reshape(-1)[0]
    assert abs(float(torch_prob) - float(onnx_prob)) < 1e-4
