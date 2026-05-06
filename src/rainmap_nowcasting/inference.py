from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from . import DEFAULT_REPO_ID
from .config import read_json, write_json
from .hf import ModelFiles, ensure_model_files
from .images import save_gif, save_prediction_frames, stack_input_frames
from .model import build_model_from_config
from .runtime import get_device


def load_model(files: ModelFiles, device_name: str = "auto") -> tuple[torch.nn.Module, dict[str, Any], torch.device]:
    config = read_json(files.config_path)
    model = build_model_from_config(config)
    device = get_device(device_name)
    state = load_file(str(files.weights_path), device=str(device))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config, device


def predict_frames(
    input_dir: str | Path,
    output_dir: str | Path,
    model_files: ModelFiles | None = None,
    repo_id: str = DEFAULT_REPO_ID,
    device_name: str = "auto",
    force_download: bool = False,
) -> dict[str, Any]:
    files = model_files or ensure_model_files(repo_id=repo_id, force_download=force_download)
    model, config, device = load_model(files, device_name=device_name)
    input_frames = int(config.get("input_frames", 6))
    image_size_raw = config.get("image_size", [128, 128])
    image_size = (int(image_size_raw[0]), int(image_size_raw[1]))

    stacked = stack_input_frames(input_dir, input_frames=input_frames, image_size=image_size)
    tensor = torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.inference_mode():
        predicted = model(tensor).squeeze(0).detach().cpu().numpy()

    predicted = np.clip(predicted, 0.0, 1.0)
    output_path = Path(output_dir)
    frame_paths = save_prediction_frames(predicted, output_path)
    gif_path = save_gif(frame_paths, output_path / "prediction.gif")
    metadata = {
        "repo_id": files.repo_id,
        "model_dir": str(files.model_dir),
        "weights": files.weights_path.name,
        "input_dir": str(Path(input_dir)),
        "output_dir": str(output_path),
        "frame_paths": [str(path) for path in frame_paths],
        "gif_path": str(gif_path),
        "warning": "Demo model only. Not for operational weather forecasting.",
    }
    write_json(output_path / "prediction_metadata.json", metadata)
    return metadata
