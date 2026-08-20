from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    true_complete = y_true == 1
    pred_complete = y_pred == 1
    tp = int(np.sum(true_complete & pred_complete))
    tn = int(np.sum(~true_complete & ~pred_complete))
    fp = int(np.sum(~true_complete & pred_complete))
    fn = int(np.sum(true_complete & ~pred_complete))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compute_metrics(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=np.int32)
    probs = np.asarray(probabilities, dtype=np.float64)
    preds = (probs >= threshold).astype(np.int32)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1], zero_division=0
    )
    counts = confusion_counts(labels, preds)
    incomplete_n = max(counts["tn"] + counts["fp"], 1)
    complete_n = max(counts["tp"] + counts["fn"], 1)
    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "incomplete_precision": float(precision[0]),
        "incomplete_recall": float(recall[0]),
        "incomplete_f1": float(f1[0]),
        "complete_precision": float(precision[1]),
        "complete_recall": float(recall[1]),
        "complete_f1": float(f1[1]),
        "false_complete_rate": counts["fp"] / incomplete_n,
        "false_incomplete_rate": counts["fn"] / complete_n,
        "brier": float(brier_score_loss(labels, probs)),
        "count": int(labels.size),
        **counts,
    }
    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    return metrics


def slice_metrics(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    categories: Sequence[str],
    threshold: float,
    min_count: int = 30,
) -> dict[str, dict[str, Any]]:
    labels = np.asarray(y_true)
    probs = np.asarray(probabilities)
    cats = np.asarray(categories)
    results: dict[str, dict[str, Any]] = {}
    for name in sorted(set(cats.tolist())):
        mask = cats == name
        if int(mask.sum()) < min_count:
            continue
        results[str(name)] = compute_metrics(labels[mask], probs[mask], threshold)
    return results


def selection_score(metrics: dict[str, Any], indic_f1: float | None = None) -> float:
    indic = indic_f1 if indic_f1 is not None else metrics.get("macro_f1", 0.0)
    return float(
        0.40 * metrics.get("macro_f1", 0.0)
        + 0.25 * metrics.get("complete_recall", 0.0)
        + 0.25 * metrics.get("incomplete_recall", 0.0)
        + 0.10 * indic
    )
