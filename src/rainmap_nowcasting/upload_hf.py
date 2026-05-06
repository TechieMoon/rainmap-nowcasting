from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


MODEL_CARD = """---
license: mit
pipeline_tag: image-to-image
tags:
- precipitation-nowcasting
- rainmap
- pytorch
- demo
---

# RainMap Nowcasting

This repository hosts the demo weights for
[`TechieMoon/rainmap-nowcasting`](https://github.com/TechieMoon/rainmap-nowcasting).

## Intended use

The model accepts 6 grayscale rain map frames and predicts the next 6 grayscale
frames. Pixel intensity is interpreted as `0..255 -> 0..1` rain intensity.

## Important warning

These first weights are trained on synthetic moving rain blobs only. They are
for validating installation, model download, inference, and the local client.
They are not suitable for operational weather forecasting.

## Files

- `rainmap-nowcasting-demo.safetensors`: PyTorch state dict.
- `model_config.json`: architecture, frame count, image size, and encoding.
- `training_metrics.json`: synthetic-demo training metrics.

## Example

```bash
python -m rainmap_nowcasting.predict --input-dir samples/input --output-dir outputs
```
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a trained RainMap Nowcasting bundle to Hugging Face.")
    parser.add_argument("--repo-id", required=True, help="Target Hugging Face model repo, e.g. TechieMoon/rainmap-nowcasting.")
    parser.add_argument("--model-dir", default="runs/demo", help="Directory containing weights/config/metrics.")
    parser.add_argument("--private", action="store_true", help="Create/update the model repository as private.")
    parser.add_argument("--revision", default=None, help="Optional target branch/revision.")
    return parser.parse_args()


def write_model_card(model_dir: Path) -> Path:
    path = model_dir / "README.md"
    path.write_text(MODEL_CARD, encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set. Create a new write token and set it only as an environment variable.")

    model_dir = Path(args.model_dir)
    required = [
        model_dir / "rainmap-nowcasting-demo.safetensors",
        model_dir / "model_config.json",
        model_dir / "training_metrics.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required model files: {', '.join(missing)}")

    write_model_card(model_dir)
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
