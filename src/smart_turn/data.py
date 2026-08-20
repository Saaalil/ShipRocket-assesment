from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from smart_turn.audio import decode_hf_audio
from smart_turn.augment import maybe_augment
from smart_turn.features import extract_log_mel
from smart_turn.splits import is_indic_language


def _audio_array(sample: dict[str, Any]) -> tuple[np.ndarray, int]:
    return decode_hf_audio(sample["audio"])


class TurnDataset(Dataset):
    """Index-based wrapper so audio is decoded only when a sample is requested."""

    def __init__(
        self,
        hf_dataset: Any,
        indices: Sequence[int] | None = None,
        augment: bool = False,
        indic_languages: Sequence[str] | None = None,
        upsample_indic: bool = False,
        seed: int = 42,
    ) -> None:
        self.dataset = hf_dataset
        selected = list(range(len(hf_dataset))) if indices is None else [int(i) for i in indices]
        if upsample_indic and indic_languages:
            languages = [str(value) for value in hf_dataset["language"]]
            extra = [
                index
                for index in selected
                if is_indic_language(languages[index], indic_languages)
            ]
            selected.extend(extra)
        self.indices = selected
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[self.indices[index]]
        audio, sample_rate = _audio_array(sample)
        if self.augment:
            audio = maybe_augment(audio, sample_rate, rng=self.rng)
        features = extract_log_mel(audio, sample_rate)
        label = 1 if bool(sample.get("endpoint_bool")) else 0
        return {
            "input_features": torch.from_numpy(features),
            "labels": torch.tensor(label, dtype=torch.float32),
            "language": str(sample.get("language", "unknown")),
            "dataset": str(sample.get("dataset", "unknown")),
            "midfiller": bool(sample.get("midfiller", False)),
            "endfiller": bool(sample.get("endfiller", False)),
        }


class TurnCollator:
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return {
            "input_features": torch.stack([item["input_features"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
        }
