from pathlib import Path

from PIL import Image

from rainmap_nowcasting.evaluate import evaluate_prediction_dirs


def test_evaluate_prediction_dirs(tmp_path: Path) -> None:
    pred_dir = tmp_path / "pred"
    target_dir = tmp_path / "target"
    pred_dir.mkdir()
    target_dir.mkdir()
    for index, value in enumerate([0, 128]):
        Image.new("L", (16, 16), value).save(pred_dir / f"prediction_{index:03d}.png")
        Image.new("L", (16, 16), value).save(target_dir / f"target_{index:03d}.png")

    result = evaluate_prediction_dirs(pred_dir, target_dir, thresholds=[0.5], fss_windows=[3])

    assert result["frame_count"] == 2
    assert result["continuous"]["mae"] == 0.0
    assert result["categorical"]["0.500"]["csi"] == 1.0
