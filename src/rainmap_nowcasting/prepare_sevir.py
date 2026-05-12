from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import fsspec
import h5py
import numpy as np
from PIL import Image

from .config import write_json

DEFAULT_SEVIR_VIL_URL = (
    "https://sevir.s3.amazonaws.com/data/vil/2017/SEVIR_VIL_STORMEVENTS_2017_0101_0630.h5"
)


def prepare_sevir_subset(
    output_dir: str | Path,
    source_url: str = DEFAULT_SEVIR_VIL_URL,
    image_size: tuple[int, int] = (64, 64),
    base_train: int = 12,
    base_val: int = 4,
    local_train: int = 4,
    local_val: int = 4,
    start_index: int = 0,
) -> dict:
    output = Path(output_dir)
    splits = [
        ("base/train", base_train),
        ("base/val", base_val),
        ("local/train", local_train),
        ("local/val", local_val),
    ]
    total_events = sum(count for _, count in splits)

    event_indices = list(range(start_index, start_index + total_events))
    written: dict[str, list[str]] = {split: [] for split, _ in splits}

    with fsspec.open(source_url, "rb", block_size=2**20) as handle:
        with h5py.File(handle, "r") as h5:
            vil = h5["vil"]
            ids = h5["id"]
            if start_index + total_events > vil.shape[0]:
                raise ValueError(f"Requested {total_events} events from index {start_index}, but file has {vil.shape[0]}")

            cursor = 0
            for split, count in splits:
                for _ in range(count):
                    event_index = event_indices[cursor]
                    event_id = _decode_id(ids[event_index])
                    sequence_dir = output / split / f"event_{event_index:04d}_{event_id}"
                    sequence_dir.mkdir(parents=True, exist_ok=True)
                    frames = _load_event(vil, event_index, image_size=image_size)
                    for frame_index, frame in enumerate(frames):
                        Image.fromarray(frame, mode="L").save(sequence_dir / f"frame_{frame_index:03d}.png")
                    written[split].append(str(sequence_dir))
                    cursor += 1

    metadata = {
        "dataset": "SEVIR VIL",
        "source_url": source_url,
        "note": "Real NEXRAD-derived VIL radar imagery, used here as a precipitation nowcasting proxy.",
        "image_size": list(image_size),
        "frames_per_event": 49,
        "start_index": start_index,
        "event_indices": event_indices,
        "splits": written,
    }
    write_json(output / "metadata.json", metadata)
    return metadata


def _load_event(dataset, event_index: int, image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    raw = dataset[event_index]
    if raw.ndim != 3:
        raise ValueError(f"Expected SEVIR VIL event shape [H, W, T], got {raw.shape}")
    frames = np.moveaxis(raw, -1, 0)
    resized = []
    for frame in frames:
        image = Image.fromarray(frame.astype(np.uint8), mode="L").resize((width, height), Image.Resampling.BILINEAR)
        resized.append(np.asarray(image, dtype=np.uint8))
    return np.stack(resized, axis=0)


def _decode_id(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a small real SEVIR VIL subset as PNG rain-map sequences.")
    parser.add_argument("--output-dir", default="data/sevir_mini")
    parser.add_argument("--source-url", default=DEFAULT_SEVIR_VIL_URL)
    parser.add_argument("--image-size", type=int, nargs=2, default=[64, 64], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--base-train", type=int, default=12)
    parser.add_argument("--base-val", type=int, default=4)
    parser.add_argument("--local-train", type=int, default=4)
    parser.add_argument("--local-val", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_sevir_subset(
        output_dir=args.output_dir,
        source_url=args.source_url,
        image_size=(args.image_size[0], args.image_size[1]),
        base_train=args.base_train,
        base_val=args.base_val,
        local_train=args.local_train,
        local_val=args.local_val,
        start_index=args.start_index,
    )
    print(f"Wrote SEVIR subset to {args.output_dir}")
    for split, paths in metadata["splits"].items():
        print(f"{split}: {len(paths)} sequences")


if __name__ == "__main__":
    main()
