import numpy as np
import pytest

from rainmap_nowcasting.metrics import categorical_metrics, continuous_metrics, evaluate_arrays, fractions_skill_score


def test_continuous_metrics() -> None:
    prediction = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    target = np.array([0.0, 1.0, 1.0], dtype=np.float32)

    metrics = continuous_metrics(prediction, target)

    assert metrics["mae"] == pytest.approx(1 / 6)
    assert metrics["rmse"] == pytest.approx(np.sqrt(1 / 12))
    assert metrics["bias"] == pytest.approx(-1 / 6)


def test_categorical_metrics() -> None:
    prediction = np.array([0.0, 0.7, 0.8, 0.1], dtype=np.float32)
    target = np.array([0.0, 0.6, 0.2, 0.9], dtype=np.float32)

    metrics = categorical_metrics(prediction, target, threshold=0.5)

    assert metrics["hits"] == 1
    assert metrics["misses"] == 1
    assert metrics["false_alarms"] == 1
    assert metrics["correct_negatives"] == 1
    assert metrics["csi"] == pytest.approx(1 / 3)
    assert metrics["pod"] == pytest.approx(1 / 2)
    assert metrics["far"] == pytest.approx(1 / 2)


def test_fractions_skill_score_perfect_forecast() -> None:
    target = np.zeros((8, 8), dtype=np.float32)
    target[2:5, 2:5] = 1.0

    assert fractions_skill_score(target, target, threshold=0.5, window_size=3) == pytest.approx(1.0)


def test_evaluate_arrays_groups_metrics() -> None:
    prediction = np.zeros((2, 8, 8), dtype=np.float32)
    target = np.zeros((2, 8, 8), dtype=np.float32)

    result = evaluate_arrays(prediction, target, thresholds=[0.5], fss_windows=[3])

    assert "continuous" in result
    assert "0.500" in result["categorical"]
    assert "0.500/w3" in result["fss"]
