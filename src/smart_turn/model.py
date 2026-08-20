from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import softmax
from transformers import WhisperConfig, WhisperPreTrainedModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from smart_turn.constants import WHISPER_ENCODER_POSITIONS
from smart_turn.losses import EndpointLoss


class SmartTurnModel(WhisperPreTrainedModel):
    """Whisper Tiny encoder + attention pooling + binary completion head."""

    def __init__(self, config: WhisperConfig) -> None:
        super().__init__(config)
        config.max_source_positions = min(
            int(config.max_source_positions), WHISPER_ENCODER_POSITIONS
        )
        self.encoder = WhisperEncoder(config)
        hidden_size = config.d_model
        self.pool_attention = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.post_init()

    def freeze_encoder(self, unfreeze_last_n: int = 0) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        if unfreeze_last_n <= 0:
            return
        blocks = list(self.encoder.layers)
        for block in blocks[-unfreeze_last_n:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    def forward(
        self,
        input_features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden_states = self.encoder(input_features=input_features).last_hidden_state
        attention = softmax(self.pool_attention(hidden_states), dim=1)
        pooled = torch.sum(hidden_states * attention, dim=1)
        logits = self.classifier(pooled)
        probabilities = torch.sigmoid(logits)
        output: dict[str, torch.Tensor] = {"logits": logits, "probabilities": probabilities}
        if labels is not None:
            output["loss"] = EndpointLoss()(logits, labels)
        return output


def load_pretrained_turn_model(base_model: str) -> SmartTurnModel:
    return SmartTurnModel.from_pretrained(
        base_model,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
