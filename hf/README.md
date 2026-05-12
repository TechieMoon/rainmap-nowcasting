---
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

The first real-data smoke benchmark uses a small SEVIR VIL subset. These weights
are still not suitable for operational weather forecasting; they are intended to
validate training, fine-tuning, benchmarking, and client distribution workflows.

## Files

- `rainmap-nowcasting-sevir-finetuned.safetensors`: PyTorch state dict.
- `model_config.json`: architecture, frame count, image size, and encoding.
- `training_metrics.json`: training metrics.
- `benchmark_results.json`: small-subset benchmark results when uploaded.

## Example

```bash
python -m rainmap_nowcasting.predict --input-dir samples/input --output-dir outputs
```
