from __future__ import annotations

import torch
from transformers import WhisperConfig

from smart_turn.model import SmartTurnModel
from smart_turn.splits import grouped_indices


def test_tiny_forward_and_freeze() -> None:
    config = WhisperConfig(
        vocab_size=100,
        num_mel_bins=80,
        encoder_layers=2,
        encoder_attention_heads=4,
        decoder_layers=1,
        decoder_attention_heads=4,
        d_model=32,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_source_positions=10,
        num_hidden_layers=2,
    )
    model = SmartTurnModel(config)
    model.freeze_encoder(unfreeze_last_n=1)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    frozen = [name for name, param in model.named_parameters() if not param.requires_grad]
    assert any("classifier" in name for name in trainable)
    assert any("encoder" in name for name in frozen)
    features = torch.zeros(2, 80, 20)
    out = model(features)
    assert out["probabilities"].shape == (2, 1)
    assert torch.all((out["probabilities"] >= 0) & (out["probabilities"] <= 1))


def test_grouped_split_is_disjoint() -> None:
    ids = [f"spk{i}-{j}" for i in range(10) for j in range(4)]
    sources = ["liva"] * len(ids)
    labels = [i % 2 for i in range(len(ids))]
    train, valid = grouped_indices(ids, sources, labels, val_fraction=0.2, seed=0)
    assert set(train).isdisjoint(set(valid))
    assert len(train) + len(valid) == len(ids)
