from __future__ import annotations

import itertools

import numpy as np


def _f1_at_threshold(labels, probabilities, threshold):
    labels = np.asarray(labels).astype(np.int32)
    probabilities = np.asarray(probabilities).astype(np.float32)
    preds = (probabilities >= threshold).astype(np.int32)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall


def best_f1_threshold(labels, probabilities, thresholds=None):
    thresholds = thresholds if thresholds is not None else np.linspace(0.0, 1.0, 101)
    best = {"f1": -1.0, "precision": 0.0, "recall": 0.0, "threshold": 0.5}
    for threshold in thresholds:
        f1, precision, recall = _f1_at_threshold(labels, probabilities, float(threshold))
        if f1 > best["f1"]:
            best = {
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "threshold": float(threshold),
            }
    return best


def weight_grid(step: float = 0.1):
    values = np.round(np.arange(step, 1.0 + step / 2, step), 10)
    for w1, w2 in itertools.product(values, values):
        w3 = round(1.0 - float(w1) - float(w2), 10)
        if w3 < -1e-9:
            continue
        if abs(float(w1) + float(w2) + w3 - 1.0) <= 1e-9:
            yield (float(w1), float(w2), float(w3))


def grid_search_weights(labels, branch_probabilities, step: float = 0.1, thresholds=None):
    p1, p2, p3 = [np.asarray(p, dtype=np.float32) for p in branch_probabilities]
    labels = np.asarray(labels).astype(np.int32)
    best = {
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "threshold": 0.5,
        "weights": (1 / 3, 1 / 3, 1 / 3),
    }
    checked = 0
    for weights in weight_grid(step=step):
        checked += 1
        combined = weights[0] * p1 + weights[1] * p2 + weights[2] * p3
        result = best_f1_threshold(labels, combined, thresholds=thresholds)
        if result["f1"] > best["f1"]:
            best = dict(result)
            best["weights"] = weights
    best["checked_weight_combinations"] = checked
    return best


__all__ = ["best_f1_threshold", "grid_search_weights", "weight_grid"]
