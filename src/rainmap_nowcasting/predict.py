from __future__ import annotations

import argparse
from pathlib import Path

from . import DEFAULT_REPO_ID
from .hf import ModelFiles
from .inference import predict_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict future rain map frames from an image sequence.")
    parser.add_argument("--input-dir", required=True, help="Directory containing at least 6 input images.")
    parser.add_argument("--output-dir", required=True, help="Directory where prediction PNG/GIF files are written.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face model repo id.")
    parser.add_argument("--model-dir", default=None, help="Use a local model directory instead of downloading.")
    parser.add_argument("--weights", default="rainmap-nowcasting-demo.safetensors", help="Weights filename in model-dir.")
    parser.add_argument("--config", default="model_config.json", help="Model config filename in model-dir.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--force-download", action="store_true", help="Download the model even if cached files exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_files = None
    if args.model_dir:
        model_dir = Path(args.model_dir)
        model_files = ModelFiles(
            model_dir=model_dir,
            weights_path=model_dir / args.weights,
            config_path=model_dir / args.config,
            repo_id="local",
        )
    metadata = predict_frames(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_files=model_files,
        repo_id=args.repo_id,
        device_name=args.device,
        force_download=args.force_download,
    )
    print(f"Wrote {len(metadata['frame_paths'])} frames and {metadata['gif_path']}")


if __name__ == "__main__":
    main()
