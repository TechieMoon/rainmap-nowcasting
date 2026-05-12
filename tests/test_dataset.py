from pathlib import Path

import pytest
from PIL import Image

from rainmap_nowcasting.data import RainMapSequenceDataset
from rainmap_nowcasting.config import read_train_config


def _write_frame(path: Path, value: int) -> None:
    Image.new("L", (16, 16), value).save(path)


def test_dataset_sorts_frames_and_builds_windows(tmp_path: Path) -> None:
    seq = tmp_path / "train" / "sequence_0001"
    seq.mkdir(parents=True)
    for name, value in [("frame_10.png", 10), ("frame_2.png", 2), ("frame_1.png", 1), ("frame_11.png", 11)]:
        _write_frame(seq / name, value)

    dataset = RainMapSequenceDataset(tmp_path / "train", input_frames=2, target_frames=1, image_size=(8, 8))

    assert len(dataset) == 2
    x, y = dataset[0]
    assert x.shape == (2, 8, 8)
    assert y.shape == (1, 8, 8)
    assert float(x[0, 0, 0]) == pytest.approx(1 / 255)
    assert float(x[1, 0, 0]) == pytest.approx(2 / 255)
    assert float(y[0, 0, 0]) == pytest.approx(10 / 255)


def test_train_config_reads_resume_from(tmp_path: Path) -> None:
    config_path = tmp_path / "fine_tune.yaml"
    config_path.write_text(
        "\n".join(
            [
                "train_dir: data/local/train",
                "val_dir: data/local/val",
                "resume_from: runs/base/model.safetensors",
            ]
        ),
        encoding="utf-8",
    )

    config = read_train_config(config_path)

    assert str(config.resume_from).replace("\\", "/") == "runs/base/model.safetensors"
