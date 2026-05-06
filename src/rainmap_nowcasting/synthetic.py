from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def generate_sequence(
    output_dir: str | Path,
    sequence_index: int,
    frame_count: int = 18,
    image_size: tuple[int, int] = (128, 128),
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed + sequence_index)
    height, width = image_size
    yy, xx = np.mgrid[0:height, 0:width]
    blobs = []
    for _ in range(rng.integers(2, 5)):
        blobs.append(
            {
                "cx": rng.uniform(width * 0.15, width * 0.85),
                "cy": rng.uniform(height * 0.15, height * 0.85),
                "vx": rng.uniform(-2.0, 2.0),
                "vy": rng.uniform(-2.0, 2.0),
                "sigma": rng.uniform(8.0, 20.0),
                "amp": rng.uniform(0.35, 1.0),
                "growth": rng.uniform(-0.025, 0.025),
            }
        )

    sequence_dir = Path(output_dir) / f"sequence_{sequence_index:04d}"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for frame_index in range(frame_count):
        frame = np.zeros((height, width), dtype=np.float32)
        for blob in blobs:
            cx = blob["cx"] + blob["vx"] * frame_index
            cy = blob["cy"] + blob["vy"] * frame_index
            amp = np.clip(blob["amp"] + blob["growth"] * frame_index, 0.05, 1.0)
            sigma = blob["sigma"] * (1.0 + 0.1 * np.sin(frame_index / 3.0))
            distance = ((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2)
            frame += amp * np.exp(-distance)
        frame += rng.normal(0.0, 0.015, size=frame.shape).astype(np.float32)
        frame = np.clip(frame, 0.0, 1.0)
        image = Image.fromarray((frame * 255).astype(np.uint8), mode="L")
        image.save(sequence_dir / f"frame_{frame_index:03d}.png")


def generate_dataset(
    output_dir: str | Path,
    train_sequences: int = 32,
    val_sequences: int = 8,
    frame_count: int = 18,
    image_size: tuple[int, int] = (128, 128),
    seed: int = 42,
) -> None:
    root = Path(output_dir)
    for split, count, offset in (("train", train_sequences, 0), ("val", val_sequences, train_sequences)):
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            generate_sequence(
                split_dir,
                sequence_index=index,
                frame_count=frame_count,
                image_size=image_size,
                seed=seed + offset,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic moving rain-map blobs for demo training.")
    parser.add_argument("--output-dir", default="data/demo", help="Dataset root to write train/val sequences into.")
    parser.add_argument("--train-sequences", type=int, default=32)
    parser.add_argument("--val-sequences", type=int, default=8)
    parser.add_argument("--frames", type=int, default=18)
    parser.add_argument("--image-size", type=int, nargs=2, default=[128, 128], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        output_dir=args.output_dir,
        train_sequences=args.train_sequences,
        val_sequences=args.val_sequences,
        frame_count=args.frames,
        image_size=(args.image_size[0], args.image_size[1]),
        seed=args.seed,
    )
    print(f"Wrote synthetic dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
