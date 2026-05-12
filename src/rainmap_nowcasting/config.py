from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainConfig:
    train_dir: Path
    val_dir: Path
    output_dir: Path
    input_frames: int = 6
    target_frames: int = 6
    image_size: tuple[int, int] = (128, 128)
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 1e-3
    base_channels: int = 32
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    amp: bool = True
    checkpoint_name: str = "rainmap-nowcasting-demo.safetensors"
    resume_from: Path | None = None
    dataset_name: str = "custom-rainmap"
    model_warning: str = "Not for operational weather forecasting unless validated on real local data."


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return loaded


def read_train_config(path: str | Path) -> TrainConfig:
    data = read_yaml(path)
    image_size = data.get("image_size", [128, 128])
    if len(image_size) != 2:
        raise ValueError("image_size must contain [height, width]")
    return TrainConfig(
        train_dir=Path(data["train_dir"]),
        val_dir=Path(data["val_dir"]),
        output_dir=Path(data.get("output_dir", "runs/demo")),
        input_frames=int(data.get("input_frames", 6)),
        target_frames=int(data.get("target_frames", 6)),
        image_size=(int(image_size[0]), int(image_size[1])),
        batch_size=int(data.get("batch_size", 8)),
        epochs=int(data.get("epochs", 3)),
        learning_rate=float(data.get("learning_rate", 1e-3)),
        base_channels=int(data.get("base_channels", 32)),
        num_workers=int(data.get("num_workers", 0)),
        seed=int(data.get("seed", 42)),
        device=str(data.get("device", "auto")),
        amp=bool(data.get("amp", True)),
        checkpoint_name=str(data.get("checkpoint_name", "rainmap-nowcasting-demo.safetensors")),
        resume_from=Path(data["resume_from"]) if data.get("resume_from") else None,
        dataset_name=str(data.get("dataset_name", "custom-rainmap")),
        model_warning=str(
            data.get("model_warning", "Not for operational weather forecasting unless validated on real local data.")
        ),
    )


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def read_json(path: str | Path) -> dict[str, Any]:
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return loaded
