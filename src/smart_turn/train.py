from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, WhisperFeatureExtractor

from smart_turn.config import load_experiment_config
from smart_turn.constants import MAX_AUDIO_SECONDS
from smart_turn.data import TurnCollator, TurnDataset
from smart_turn.evaluate import compute_metrics
from smart_turn.model import SmartTurnModel
from smart_turn.splits import grouped_indices, is_indic_language, keep_language, language_allowlist


def _load_hf_dataset(
    name: str,
    max_samples: int | None = None,
    keep_languages: list[str] | None = None,
    indic_languages: list[str] | None = None,
    min_indic_fraction: float = 0.2,
):
    """Stream rows so Colab never materializes the full 41 GB Arrow cache.

    Language is a column inside mixed parquet shards. There are no separate
    Hindi/English/Hinglish files, and Hinglish is not labeled.
    """
    from datasets import Dataset, load_dataset

    stream = load_dataset(name, split="train", streaming=True)
    allowed = language_allowlist(keep_languages or [])
    indic = {item.lower() for item in (indic_languages or [])}
    limit = int(max_samples) if max_samples else None
    min_indic = int((limit or 0) * min_indic_fraction) if indic and limit else 0
    rows: list[dict[str, Any]] = []
    indic_count = 0
    non_indic_budget = None if limit is None else max(0, limit - min_indic)

    for row in stream:
        language = str(row.get("language", ""))
        if allowed and not keep_language(language, allowed):
            continue
        is_indic = bool(indic) and is_indic_language(language, list(indic))
        if limit is not None and len(rows) >= limit:
            break
        if (
            limit is not None
            and not is_indic
            and non_indic_budget is not None
            and (len(rows) - indic_count) >= non_indic_budget
            and indic_count < min_indic
        ):
            continue
        rows.append(row)
        indic_count += int(is_indic)
        if limit is not None and len(rows) >= limit and indic_count >= min_indic:
            break
    if not rows:
        raise RuntimeError(f"No rows loaded from {name} with filters {sorted(allowed)}")
    return Dataset.from_list(rows)


def build_model(config: dict[str, Any]) -> SmartTurnModel:
    model = SmartTurnModel.from_pretrained(
        config["base_model"],
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    if config.get("freeze_encoder", True):
        model.freeze_encoder(unfreeze_last_n=int(config.get("unfreeze_encoder_layers", 0)))
    return model


def prepare_splits(config: dict[str, Any]) -> tuple[TurnDataset, TurnDataset]:
    max_samples = config.get("max_train_samples")
    keep_languages = list(config.get("english_codes", [])) + list(config.get("indic_languages", []))
    raw = _load_hf_dataset(
        config["train_dataset"],
        max_samples=max_samples,
        keep_languages=keep_languages or None,
        indic_languages=list(config.get("indic_languages", [])),
        min_indic_fraction=float(config.get("min_indic_fraction", 0.2)),
    )
    ids = [str(value) for value in raw["id"]]
    sources = [str(value) for value in raw["dataset"]]
    labels = [1 if value else 0 for value in raw["endpoint_bool"]]
    train_idx, val_idx = grouped_indices(
        ids,
        sources,
        labels,
        val_fraction=float(config.get("val_fraction", 0.1)),
        seed=int(config.get("seed", 42)),
    )
    train_ds = TurnDataset(
        raw,
        indices=train_idx,
        augment=True,
        indic_languages=config.get("indic_languages", []),
        upsample_indic=bool(config.get("upsample_indic", False)),
        seed=int(config.get("seed", 42)),
    )
    val_ds = TurnDataset(raw, indices=val_idx, augment=False)
    return train_ds, val_ds


def train_from_config(config_path: str) -> Path:
    config = load_experiment_config(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(config)
    train_ds, val_ds = prepare_splits(config)
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(config.get("train_batch_size", 8)),
        per_device_eval_batch_size=int(config.get("eval_batch_size", 16)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 16)),
        num_train_epochs=float(config.get("num_epochs", 1)),
        learning_rate=float(config.get("learning_rate", 5e-5)),
        warmup_ratio=float(config.get("warmup_ratio", 0.08)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        fp16=bool(config.get("fp16", True)),
        eval_strategy="steps",
        eval_steps=int(config.get("eval_steps", 400)),
        save_steps=int(config.get("save_steps", 400)),
        logging_steps=int(config.get("logging_steps", 50)),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        save_total_limit=2,
    )

    def compute_metrics_hf(eval_pred) -> dict[str, float]:
        logits = eval_pred.predictions
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probs = 1.0 / (1.0 + np.exp(-np.asarray(logits).reshape(-1)))
        labels = np.asarray(eval_pred.label_ids).reshape(-1)
        return compute_metrics(labels, probs, float(config.get("threshold", 0.5)))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=TurnCollator(),
        compute_metrics=compute_metrics_hf,
    )
    resume = config.get("resume_from")
    trainer.train(resume_from_checkpoint=resume if resume else None)
    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    WhisperFeatureExtractor(chunk_length=MAX_AUDIO_SECONDS).save_pretrained(str(final_dir))
    return final_dir


def predict_loader(model: SmartTurnModel, dataset: TurnDataset, batch_size: int = 8) -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=TurnCollator())
    probs: list[np.ndarray] = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            features = batch["input_features"].to(device)
            output = model(features)
            probs.append(output["probabilities"].detach().cpu().numpy().reshape(-1))
    return np.concatenate(probs) if probs else np.array([], dtype=np.float32)
