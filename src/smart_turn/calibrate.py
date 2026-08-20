from __future__ import annotations

import numpy as np

from smart_turn.evaluate import compute_metrics, selection_score


def sweep_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    indic_mask: np.ndarray | None = None,
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.05,
) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    best_threshold = 0.5
    best_score = -1.0
    for raw in np.arange(start, stop + 1e-9, step):
        threshold = float(round(raw, 2))
        metrics = compute_metrics(y_true, probabilities, threshold)
        indic_f1 = None
        if indic_mask is not None and indic_mask.any():
            indic_f1 = compute_metrics(
                y_true[indic_mask], probabilities[indic_mask], threshold
            )["macro_f1"]
        score = selection_score(metrics, indic_f1)
        rows.append({"threshold": threshold, "selection_score": score, **metrics})
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold, rows
