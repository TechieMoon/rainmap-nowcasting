from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainConfig, read_train_config, write_json
from .data import RainMapSequenceDataset
from .model import RainMapUNet
from .runtime import get_device, seed_everything


def make_loader(dataset: RainMapSequenceDataset, config: TrainConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            batch_size = x.shape[0]
            total_loss += float(loss_fn(pred, y).detach().cpu()) * batch_size
            total_mae += float(torch.mean(torch.abs(pred - y)).detach().cpu()) * batch_size
            total_count += batch_size
    return {
        "mse": total_loss / max(total_count, 1),
        "mae": total_mae / max(total_count, 1),
    }


def train(config: TrainConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    device = get_device(config.device)
    train_dataset = RainMapSequenceDataset(
        config.train_dir,
        input_frames=config.input_frames,
        target_frames=config.target_frames,
        image_size=config.image_size,
    )
    val_dataset = RainMapSequenceDataset(
        config.val_dir,
        input_frames=config.input_frames,
        target_frames=config.target_frames,
        image_size=config.image_size,
    )
    train_loader = make_loader(train_dataset, config, shuffle=True)
    val_loader = make_loader(val_dataset, config, shuffle=False)

    model = RainMapUNet(
        input_frames=config.input_frames,
        target_frames=config.target_frames,
        base_channels=config.base_channels,
    ).to(device)
    if config.resume_from:
        state = load_file(str(config.resume_from), device=str(device))
        model.load_state_dict(state)
        print(f"Loaded checkpoint for fine-tuning: {config.resume_from}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")

    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                pred = model(x)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = x.shape[0]
            running_loss += float(loss.detach().cpu()) * batch_size
            running_count += batch_size
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        epoch_metrics = {
            "epoch": epoch,
            "train_mse": running_loss / max(running_count, 1),
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
        }
        history.append(epoch_metrics)
        print(
            f"epoch={epoch} train_mse={epoch_metrics['train_mse']:.6f} "
            f"val_mse={epoch_metrics['val_mse']:.6f} val_mae={epoch_metrics['val_mae']:.6f}"
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = config.output_dir / config.checkpoint_name
    save_file(model.state_dict(), str(weights_path))
    model_config = {
        "model_type": "RainMapUNet",
        "input_frames": config.input_frames,
        "target_frames": config.target_frames,
        "image_size": list(config.image_size),
        "base_channels": config.base_channels,
        "pixel_encoding": "grayscale_0_255_to_0_1",
        "rain_scale": {
            "min_mm_per_hour": 0.0,
            "max_mm_per_hour": 50.0,
            "description": "Linear placeholder scale for MVP demos; calibrate with real data before use.",
        },
        "weights": weights_path.name,
        "dataset_name": config.dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "warning": config.model_warning,
    }
    write_json(config.output_dir / "model_config.json", model_config)
    metrics = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "device": str(device),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "history": history,
        "best_val_mse": min(item["val_mse"] for item in history),
        "warning": config.model_warning,
        "resume_from": str(config.resume_from) if config.resume_from else None,
    }
    write_json(config.output_dir / "training_metrics.json", metrics)
    return {"weights_path": str(weights_path), "model_config": model_config, "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the rain map nowcasting model.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to a training YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train(read_train_config(args.config))
    print(f"Saved weights to {result['weights_path']}")


if __name__ == "__main__":
    main()
