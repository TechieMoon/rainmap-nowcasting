---
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
