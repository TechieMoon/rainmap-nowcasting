from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import write_json
from .hf import model_files_from_dir
from .images import list_image_files, read_grayscale_image
from .inference import load_model
from .metrics import evaluate_arrays


def benchmark(
    sequences_dir: str | Path,
    output_dir: str | Path,
    models: dict[str, str],
    thresholds: list[float],
    fss_windows: list[int],
    input_frames: int = 6,
    target_frames: int = 6,
    image_size: tuple[int, int] = (64, 64),
    device_name: str = "auto",
) -> dict:
    sequence_dirs = sorted([p for p in Path(sequences_dir).iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    if not sequence_dirs:
        raise ValueError(f"No sequence directories found in {sequences_dir}")

    target_stack = _load_targets(sequence_dirs, input_frames, target_frames, image_size)
    results = {
        "dataset": str(Path(sequences_dir)),
        "sequence_count": len(sequence_dirs),
        "input_frames": input_frames,
        "target_frames": target_frames,
        "image_size": list(image_size),
        "thresholds": thresholds,
        "fss_windows": fss_windows,
        "models": {},
    }

    persistence = _predict_persistence(sequence_dirs, input_frames, target_frames, image_size)
    results["models"]["persistence"] = evaluate_arrays(persistence, target_stack, thresholds, fss_windows)

    for name, model_dir in models.items():
        predictions = _predict_model(sequence_dirs, model_dir, input_frames, image_size, device_name)
        results["models"][name] = evaluate_arrays(predictions, target_stack, thresholds, fss_windows)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "benchmark_results.json", results)
    (output / "benchmark_results.md").write_text(format_markdown(results), encoding="utf-8")
    return results


def _load_window(
    sequence_dir: Path,
    input_frames: int,
    target_frames: int,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    paths = list_image_files(sequence_dir)
    required = input_frames + target_frames
    if len(paths) < required:
        raise ValueError(f"Need at least {required} frames in {sequence_dir}; found {len(paths)}")
    frames = [read_grayscale_image(path, image_size) for path in paths[:required]]
    input_stack = np.stack(frames[:input_frames], axis=0)
    target_stack = np.stack(frames[input_frames:], axis=0)
    return input_stack, target_stack


def _load_targets(
    sequence_dirs: list[Path],
    input_frames: int,
    target_frames: int,
    image_size: tuple[int, int],
) -> np.ndarray:
    return np.concatenate(
        [_load_window(seq, input_frames, target_frames, image_size)[1] for seq in sequence_dirs],
        axis=0,
    )


def _predict_persistence(
    sequence_dirs: list[Path],
    input_frames: int,
    target_frames: int,
    image_size: tuple[int, int],
) -> np.ndarray:
    predictions = []
    for seq in sequence_dirs:
        x, _ = _load_window(seq, input_frames, target_frames, image_size)
        predictions.append(np.repeat(x[-1][None, :, :], target_frames, axis=0))
    return np.concatenate(predictions, axis=0)


def _predict_model(
    sequence_dirs: list[Path],
    model_dir: str | Path,
    input_frames: int,
    image_size: tuple[int, int],
    device_name: str,
) -> np.ndarray:
    files = model_files_from_dir(model_dir)
    model, config, device = load_model(files, device_name=device_name)
    config_image_size = tuple(int(value) for value in config.get("image_size", image_size))
    if config_image_size != image_size:
        image_size = config_image_size
    predictions = []
    with torch.inference_mode():
        for seq in sequence_dirs:
            x, _ = _load_window(seq, input_frames, int(config.get("target_frames", 6)), image_size)
            tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(tensor).squeeze(0).detach().cpu().numpy()
            predictions.append(np.clip(pred, 0.0, 1.0))
    return np.concatenate(predictions, axis=0)


def format_markdown(results: dict) -> str:
    thresholds = [str(value) for value in results["thresholds"]]
    primary_threshold = f"{float(thresholds[1] if len(thresholds) > 1 else thresholds[0]):.3f}"
    rows = [
        "# Real-Data Benchmark Results",
        "",
        f"- Dataset: `{results['dataset']}`",
        f"- Sequences: {results['sequence_count']}",
        f"- Frames: {results['input_frames']} input -> {results['target_frames']} target",
        f"- Primary threshold: `{primary_threshold}`",
        "",
        "| Model | MAE | RMSE | Bias | CSI | POD | FAR | HSS | ETS | FSS w15 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in results["models"].items():
        continuous = metrics["continuous"]
        categorical = metrics["categorical"].get(primary_threshold)
        if categorical is None:
            categorical = next(iter(metrics["categorical"].values()))
            primary_threshold = next(iter(metrics["categorical"].keys()))
        fss_key = f"{primary_threshold}/w15"
        fss_value = metrics["fss"].get(fss_key, next(iter(metrics["fss"].values())))
        rows.append(
            "| {name} | {mae:.4f} | {rmse:.4f} | {bias:.4f} | {csi:.4f} | {pod:.4f} | {far:.4f} | {hss:.4f} | {ets:.4f} | {fss:.4f} |".format(
                name=name,
                mae=continuous["mae"],
                rmse=continuous["rmse"],
                bias=continuous["bias"],
                csi=categorical["csi"],
                pod=categorical["pod"],
                far=categorical["far"],
                hss=categorical["hss"],
                ets=categorical["ets"],
                fss=fss_value,
            )
        )
    rows.extend(
        [
            "",
            "These are small-subset smoke benchmark results, not operational forecasting claims.",
            "",
        ]
    )
    return "\n".join(rows)


def parse_model_args(values: list[str]) -> dict[str, str]:
    models = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--model values must use name=path")
        name, path = value.split("=", 1)
        models[name] = path
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark persistence and model predictions on rain-map sequences.")
    parser.add_argument("--sequences-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/real_benchmark")
    parser.add_argument("--model", action="append", default=[], help="Model as name=path. Can be repeated.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--fss-windows", type=int, nargs="+", default=[5, 15])
    parser.add_argument("--image-size", type=int, nargs=2, default=[64, 64], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--input-frames", type=int, default=6)
    parser.add_argument("--target-frames", type=int, default=6)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = benchmark(
        sequences_dir=args.sequences_dir,
        output_dir=args.output_dir,
        models=parse_model_args(args.model),
        thresholds=args.thresholds,
        fss_windows=args.fss_windows,
        input_frames=args.input_frames,
        target_frames=args.target_frames,
        image_size=(args.image_size[0], args.image_size[1]),
        device_name=args.device,
    )
    print(format_markdown(results))


if __name__ == "__main__":
    main()
