from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def grouped_indices(
    ids: Sequence[str],
    sources: Sequence[str],
    labels: Sequence[int],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic grouped split. Groups are source + id prefix, stratified by label."""
    if not (len(ids) == len(sources) == len(labels)):
        raise ValueError("ids, sources, and labels must be the same length")
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    group_label: dict[str, int] = {}
    for index, (sample_id, source, label) in enumerate(zip(ids, sources, labels, strict=True)):
        prefix = str(sample_id).split("-")[0]
        group = f"{source}:{prefix}"
        groups.setdefault(group, []).append(index)
        group_label[group] = int(label)

    train: list[int] = []
    valid: list[int] = []
    for label_value in (0, 1):
        label_groups = [name for name, value in group_label.items() if value == label_value]
        order = rng.permutation(len(label_groups))
        split_at = max(1, int(round(len(label_groups) * val_fraction))) if label_groups else 0
        for rank, group_index in enumerate(order):
            members = groups[label_groups[group_index]]
            if rank < split_at:
                valid.extend(members)
            else:
                train.extend(members)
    return np.array(sorted(train), dtype=np.int64), np.array(sorted(valid), dtype=np.int64)


def is_indic_language(language: str, indic_languages: Sequence[str]) -> bool:
    code = str(language).strip().lower()
    return code in {item.lower() for item in indic_languages}
