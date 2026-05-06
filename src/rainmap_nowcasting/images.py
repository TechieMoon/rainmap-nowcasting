from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def natural_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def list_image_files(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        if root.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image file: {root}")
        return [root]

    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=natural_key)


def read_grayscale_image(path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    with Image.open(path) as image:
        gray = image.convert("L").resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(gray, dtype=np.float32) / 255.0


def frame_to_uint8(frame: np.ndarray) -> np.ndarray:
    return np.clip(frame * 255.0, 0, 255).astype(np.uint8)


def save_prediction_frames(frames: np.ndarray, output_dir: str | Path, prefix: str = "prediction") -> list[Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for index, frame in enumerate(frames):
        image = Image.fromarray(frame_to_uint8(frame), mode="L")
        path = target_dir / f"{prefix}_{index:03d}.png"
        image.save(path)
        saved.append(path)
    return saved


def save_gif(frame_paths: Iterable[str | Path], output_path: str | Path, duration_ms: int = 350) -> Path:
    paths = [Path(path) for path in frame_paths]
    if not paths:
        raise ValueError("Cannot create GIF without frames")
    images = [Image.open(path).convert("L") for path in paths]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    first, *rest = images
    first.save(target, save_all=True, append_images=rest, duration=duration_ms, loop=0)
    for image in images:
        image.close()
    return target


def stack_input_frames(input_dir: str | Path, input_frames: int, image_size: tuple[int, int]) -> np.ndarray:
    paths = list_image_files(input_dir)
    if len(paths) < input_frames:
        raise ValueError(f"Need at least {input_frames} images, found {len(paths)} in {input_dir}")
    selected = paths[-input_frames:]
    return np.stack([read_grayscale_image(path, image_size) for path in selected], axis=0)
