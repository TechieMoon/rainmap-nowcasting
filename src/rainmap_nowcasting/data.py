from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .images import list_image_files, read_grayscale_image


@dataclass(frozen=True)
class WindowRef:
    sequence_dir: Path
    start: int
    frames: tuple[Path, ...]


class RainMapSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sliding-window dataset for grayscale rain map sequences."""

    def __init__(
        self,
        root_dir: str | Path,
        input_frames: int = 6,
        target_frames: int = 6,
        image_size: tuple[int, int] = (128, 128),
    ) -> None:
        self.root_dir = Path(root_dir)
        self.input_frames = input_frames
        self.target_frames = target_frames
        self.image_size = image_size
        self.window_size = input_frames + target_frames
        self.windows = self._build_windows()
        if not self.windows:
            raise ValueError(
                f"No valid windows found in {self.root_dir}. "
                f"Each sequence needs at least {self.window_size} images."
            )

    def _sequence_dirs(self) -> list[Path]:
        direct_frames = list_image_files(self.root_dir) if self.root_dir.exists() else []
        if direct_frames:
            return [self.root_dir]
        return sorted([p for p in self.root_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower())

    def _build_windows(self) -> list[WindowRef]:
        windows: list[WindowRef] = []
        for sequence_dir in self._sequence_dirs():
            frames = tuple(list_image_files(sequence_dir))
            if len(frames) < self.window_size:
                continue
            for start in range(0, len(frames) - self.window_size + 1):
                windows.append(WindowRef(sequence_dir=sequence_dir, start=start, frames=frames[start : start + self.window_size]))
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[index]
        loaded = [read_grayscale_image(path, self.image_size) for path in window.frames]
        tensor = torch.from_numpy(np.stack(loaded, axis=0)).float()
        x = tensor[: self.input_frames]
        y = tensor[self.input_frames :]
        return x, y
