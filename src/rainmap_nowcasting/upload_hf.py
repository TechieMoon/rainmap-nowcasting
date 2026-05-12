from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

from .config import read_json, write_json

MODEL_CARD_TEMPLATE = """---
license: mit
pipeline_tag: image-to-image
tags:
- precipitation-nowcasting
- rainmap
- pytorch
- sevir
- fine-tuning
---

# RainMap Nowcasting

This repository hosts model weights for
[`TechieMoon/rainmap-nowcasting`](https://github.com/TechieMoon/rainmap-nowcasting).

## Intended use

The model accepts 6 grayscale rain map frames and predicts the next 6 grayscale
frames. Pixel intensity is interpreted as `0..255 -> 0..1` rain intensity.

## Important warning

{warning}

## Model

- Dataset: `{dataset_name}`
- Weights: `{weights}`
- Input frames: `{input_frames}`
- Target frames: `{target_frames}`
- Image size: `{image_size}`

## Files

- `{weights}`: PyTorch state dict.
- `model_config.json`: architecture, frame count, image size, and encoding.
- `training_metrics.json`: training metrics.

{benchmark_section}

## Example

```bash
python -m rainmap_nowcasting.predict --input-dir samples/input --output-dir outputs
```
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a trained RainMap Nowcasting bundle to Hugging Face.")
    parser.add_argument("--repo-id", required=True, help="Target Hugging Face model repo, e.g. TechieMoon/rainmap-nowcasting.")
    parser.add_argument("--model-dir", default="runs/demo", help="Directory containing weights/config/metrics.")
    parser.add_argument("--benchmark-json", default=None, help="Optional benchmark_results.json to include in upload/card.")
    parser.add_argument("--private", action="store_true", help="Create/update the model repository as private.")
    parser.add_argument("--revision", default=None, help="Optional target branch/revision.")
    return parser.parse_args()


def write_model_card(model_dir: Path, benchmark_json: str | None = None) -> Path:
    config = read_json(model_dir / "model_config.json")
    metrics_path = model_dir / "training_metrics.json"
    if metrics_path.exists():
        training_metrics = read_json(metrics_path)
    else:
        training_metrics = {}
    benchmark_section = ""
    if benchmark_json:
        source = Path(benchmark_json)
        if source.exists():
            benchmark = read_json(source)
            uploaded = model_dir / "benchmark_results.json"
            write_json(uploaded, benchmark)
            benchmark_section = "\n## Benchmark\n\n" + _benchmark_markdown(benchmark) + "\n"

    path = model_dir / "README.md"
    path.write_text(
        MODEL_CARD_TEMPLATE.format(
            warning=config.get("warning", "Not for operational weather forecasting unless validated on real local data."),
            dataset_name=config.get("dataset_name", training_metrics.get("config", {}).get("dataset_name", "unknown")),
            weights=config.get("weights", "model.safetensors"),
            input_frames=config.get("input_frames", 6),
            target_frames=config.get("target_frames", 6),
            image_size=config.get("image_size", [128, 128]),
            benchmark_section=benchmark_section,
        ),
        encoding="utf-8",
    )
    return path


def _benchmark_markdown(benchmark: dict) -> str:
    if "models" not in benchmark:
        return "Benchmark JSON was uploaded with the model."
    lines = [
        f"- Dataset: `{benchmark.get('dataset', 'unknown')}`",
        f"- Sequences: {benchmark.get('sequence_count', 'unknown')}",
        "",
        "| Model | MAE | RMSE | CSI@0.3 | HSS@0.3 | ETS@0.3 | FSS@0.3/w15 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in benchmark["models"].items():
        continuous = metrics["continuous"]
        categorical = metrics["categorical"].get("0.300", next(iter(metrics["categorical"].values())))
        fss = metrics["fss"].get("0.300/w15", next(iter(metrics["fss"].values())))
        lines.append(
            f"| {name} | {continuous['mae']:.4f} | {continuous['rmse']:.4f} | "
            f"{categorical['csi']:.4f} | {categorical['hss']:.4f} | {categorical['ets']:.4f} | {fss:.4f} |"
        )
    lines.append("\nSmall-subset benchmark results are not operational forecasting claims.")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set. Create a new write token and set it only as an environment variable.")

    model_dir = Path(args.model_dir)
    config_path = model_dir / "model_config.json"
    if not config_path.exists():
        raise SystemExit(f"Missing required model config: {config_path}")
    config = read_json(config_path)
    weights = str(config.get("weights", "rainmap-nowcasting-demo.safetensors"))
    required = [model_dir / weights, config_path, model_dir / "training_metrics.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required model files: {', '.join(missing)}")

    write_model_card(model_dir, benchmark_json=args.benchmark_json)
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(model_dir),
        revision=args.revision,
        commit_message="Upload RainMap Nowcasting demo model",
    )
    print(f"Uploaded {model_dir} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
