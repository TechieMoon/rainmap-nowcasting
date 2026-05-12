from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


EPS = 1e-8


@dataclass(frozen=True)
class Contingency:
    hits: int
    misses: int
    false_alarms: int
    correct_negatives: int

    @property
    def total(self) -> int:
        return self.hits + self.misses + self.false_alarms + self.correct_negatives


def continuous_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred, truth = _validate_pair(prediction, target)
    diff = pred - truth
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    bias = float(np.mean(pred - truth))
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
    }


def contingency_table(prediction: np.ndarray, target: np.ndarray, threshold: float) -> Contingency:
    pred, truth = _validate_pair(prediction, target)
    pred_event = pred >= threshold
    true_event = truth >= threshold
    hits = int(np.logical_and(pred_event, true_event).sum())
    misses = int(np.logical_and(~pred_event, true_event).sum())
    false_alarms = int(np.logical_and(pred_event, ~true_event).sum())
    correct_negatives = int(np.logical_and(~pred_event, ~true_event).sum())
    return Contingency(
        hits=hits,
        misses=misses,
        false_alarms=false_alarms,
        correct_negatives=correct_negatives,
    )


def categorical_metrics(prediction: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float | int]:
    table = contingency_table(prediction, target, threshold)
    h = table.hits
    m = table.misses
    f = table.false_alarms
    c = table.correct_negatives
    total = max(table.total, 1)
    random_hits = ((h + m) * (h + f)) / total
    return {
        "threshold": threshold,
        "hits": h,
        "misses": m,
        "false_alarms": f,
        "correct_negatives": c,
        "csi": _safe_div(h, h + m + f),
        "pod": _safe_div(h, h + m),
        "far": _safe_div(f, h + f),
        "bias_score": _safe_div(h + f, h + m),
        "hss": _safe_div(2 * (h * c - m * f), (h + m) * (m + c) + (h + f) * (f + c)),
        "ets": _safe_div(h - random_hits, h + m + f - random_hits),
        "f1": _safe_div(2 * h, 2 * h + m + f),
    }


def fractions_skill_score(
    prediction: np.ndarray,
    target: np.ndarray,
    threshold: float,
    window_size: int,
) -> float:
    pred, truth = _validate_pair(prediction, target)
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    pred_fraction = _window_fraction(pred >= threshold, window_size)
    truth_fraction = _window_fraction(truth >= threshold, window_size)
    numerator = float(np.sum((pred_fraction - truth_fraction) ** 2))
    denominator = float(np.sum(pred_fraction**2 + truth_fraction**2))
    if denominator <= EPS:
        return 1.0
    return 1.0 - numerator / denominator


def evaluate_arrays(
    prediction: np.ndarray,
    target: np.ndarray,
    thresholds: Iterable[float] = (0.1, 0.3, 0.5),
    fss_windows: Iterable[int] = (5, 15),
) -> dict:
    pred, truth = _validate_pair(prediction, target)
    threshold_values = [float(value) for value in thresholds]
    window_values = [int(value) for value in fss_windows]
    categorical = {
        f"{threshold:.3f}": categorical_metrics(pred, truth, threshold) for threshold in threshold_values
    }
    fss = {
        f"{threshold:.3f}/w{window}": fractions_skill_score(pred, truth, threshold, window)
        for threshold in threshold_values
        for window in window_values
    }
    return {
        "continuous": continuous_metrics(pred, truth),
        "categorical": categorical,
        "fss": fss,
    }


def _validate_pair(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape} vs target {truth.shape}")
    return pred, truth


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) <= EPS:
        return 0.0
    return float(numerator / denominator)


def _window_fraction(mask: np.ndarray, window_size: int) -> np.ndarray:
    array = np.asarray(mask, dtype=np.float32)
    if array.ndim == 2:
        return _window_fraction_2d(array, window_size)
    if array.ndim == 3:
        return np.stack([_window_fraction_2d(frame, window_size) for frame in array], axis=0)
    raise ValueError("FSS expects a 2D frame or 3D frame stack")


def _window_fraction_2d(array: np.ndarray, window_size: int) -> np.ndarray:
    pad_before = window_size // 2
    pad_after = window_size - 1 - pad_before
    padded = np.pad(array, ((pad_before, pad_after), (pad_before, pad_after)), mode="constant")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    sums = (
        integral[window_size:, window_size:]
        - integral[:-window_size, window_size:]
        - integral[window_size:, :-window_size]
        + integral[:-window_size, :-window_size]
    )
    return sums / float(window_size * window_size)
