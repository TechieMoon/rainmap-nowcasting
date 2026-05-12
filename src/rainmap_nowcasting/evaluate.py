from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import write_json
from .images import list_image_files, read_grayscale_image
from .inference import predict_frames
from .metrics import evaluate_arrays


def load_frame_stack(directory: str | Path, image_size: tuple[int, int] | None = None) -> np.ndarray:
    paths = list_image_files(directory)
    if not paths:
        raise ValueError(f"No image files found in {directory}")
    if image_size is None:
        from PIL import Image

        with Image.open(paths[0]) as image:
            width, height = image.size
        image_size = (height, width)
    return np.stack([read_grayscale_image(path, image_size) for path in paths], axis=0)


def evaluate_prediction_dirs(
    prediction_dir: str | Path,
    target_dir: str | Path,
    thresholds: list[float],
    fss_windows: list[int],
    image_size: tuple[int, int] | None = None,
) -> dict:
    prediction = load_frame_stack(prediction_dir, image_size=image_size)
    target = load_frame_stack(target_dir, image_size=image_size)
    frame_count = min(prediction.shape[0], target.shape[0])
    if prediction.shape[0] != target.shape[0]:
        prediction = prediction[:frame_count]
        target = target[:frame_count]
    result = evaluate_arrays(prediction, target, thresholds=thresholds, fss_windows=fss_windows)
    result["frame_count"] = frame_count
    result["prediction_dir"] = str(Path(prediction_dir))
    result["target_dir"] = str(Path(target_dir))
    return result


def evaluate_sequence(
    sequence_dir: str | Path,
    output_dir: str | Path,
    model_dir: str | Path | None,
    thresholds: list[float],
    fss_windows: list[int],
    input_frames: int,
    target_frames: int,
    device: str,
) -> dict:
    paths = list_image_files(sequence_dir)
    required = input_frames + target_frames
    if len(paths) < required:
        raise ValueError(f"Need at least {required} frames in {sequence_dir}; found {len(paths)}")

    temp_input = Path(output_dir) / "_benchmark_input"
    temp_input.mkdir(parents=True, exist_ok=True)
    for stale in temp_input.glob("*"):
        stale.unlink()
    for path in paths[:input_frames]:
        target = temp_input / path.name
        target.write_bytes(path.read_bytes())

    prediction_output = Path(output_dir) / "predictions"
    kwargs = {"device_name": device}
    if model_dir:
        from .hf import ModelFiles

        model_root = Path(model_dir)
        kwargs["model_files"] = ModelFiles(
            model_dir=model_root,
            weights_path=model_root / "rainmap-nowcasting-demo.safetensors",
            config_path=model_root / "model_config.json",
            repo_id="local",
        )
    predict_frames(temp_input, prediction_output, **kwargs)

    target_output = Path(output_dir) / "_benchmark_target"
    target_output.mkdir(parents=True, exist_ok=True)
    for stale in target_output.glob("*"):
        stale.unlink()
    for path in paths[input_frames : input_frames + target_frames]:
        target = target_output / path.name
        target.write_bytes(path.read_bytes())

    return evaluate_prediction_dirs(
        prediction_output,
        target_output,
        thresholds=thresholds,
        fss_windows=fss_windows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rain map nowcasting predictions.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prediction-dir", help="Directory containing predicted frames.")
    mode.add_argument("--sequence-dir", help="Directory containing input+target frames for one sequence.")
    parser.add_argument("--target-dir", help="Directory containing target frames when using --prediction-dir.")
    parser.add_argument("--output-json", default=None, help="Optional path to write metrics JSON.")
    parser.add_argument("--output-dir", default="outputs/benchmark", help="Output folder for --sequence-dir mode.")
    parser.add_argument("--model-dir", default=None, help="Local model directory for --sequence-dir mode.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--fss-windows", type=int, nargs="+", default=[5, 15])
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--input-frames", type=int, default=6)
    parser.add_argument("--target-frames", type=int, default=6)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_size = tuple(args.image_size) if args.image_size else None
    if args.prediction_dir:
        if not args.target_dir:
            raise SystemExit("--target-dir is required with --prediction-dir")
        result = evaluate_prediction_dirs(
            args.prediction_dir,
            args.target_dir,
            thresholds=args.thresholds,
            fss_windows=args.fss_windows,
            image_size=image_size,
        )
    else:
        result = evaluate_sequence(
            args.sequence_dir,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            thresholds=args.thresholds,
            fss_windows=args.fss_windows,
            input_frames=args.input_frames,
            target_frames=args.target_frames,
            device=args.device,
        )

    if args.output_json:
        write_json(args.output_json, result)
    print_summary(result)


def print_summary(result: dict) -> None:
    continuous = result["continuous"]
    print(
        f"frames={result['frame_count']} "
        f"mae={continuous['mae']:.6f} rmse={continuous['rmse']:.6f} bias={continuous['bias']:.6f}"
    )
    for threshold, metrics in result["categorical"].items():
        print(
            f"threshold={threshold} csi={metrics['csi']:.4f} pod={metrics['pod']:.4f} "
            f"far={metrics['far']:.4f} hss={metrics['hss']:.4f} ets={metrics['ets']:.4f}"
        )
    for key, value in result["fss"].items():
        print(f"fss[{key}]={value:.4f}")


if __name__ == "__main__":
    main()
