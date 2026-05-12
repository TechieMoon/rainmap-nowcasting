# Fine-Tuning Guide

The recommended product/research strategy is:

```text
general base model -> local fine-tuning -> local calibration
```

This keeps one public model useful for many users while allowing local teams to
adapt it to their own radar coverage, terrain, seasonality, and rainfall bias.

## Data Layout

The current trainer expects image sequences:

```text
data/train/<sequence_id>/frame_000.png
data/train/<sequence_id>/frame_001.png
data/val/<sequence_id>/frame_000.png
```

Each sequence should contain at least 12 frames for the default setup:

```text
6 input frames + 6 target frames
```

Use natural filename ordering, such as:

```text
202605120900.png
202605120905.png
202605120910.png
```

## Base Model Training

```powershell
py -3.11 -m rainmap_nowcasting.train --config configs/train.yaml
```

For first real-data experiments, create a config that points at real train/val
folders and keeps the same interface:

```yaml
train_dir: data/real/train
val_dir: data/real/val
output_dir: runs/base-real-v1
input_frames: 6
target_frames: 6
image_size: [128, 128]
batch_size: 8
epochs: 20
learning_rate: 0.001
base_channels: 32
device: auto
```

## Local Fine-Tuning

Fine-tuning uses the same trainer with `resume_from` pointing to a base model.
Use a smaller learning rate than base training:

```powershell
py -3.11 -m rainmap_nowcasting.train --config configs/fine_tune.yaml
```

The recommended experiment naming convention is:

```text
runs/base-real-v1
runs/finetune-korea-seoul-v1
runs/finetune-user-region-v1
```

The fine-tuning config must keep the same `input_frames`, `target_frames`, and
`base_channels` as the base checkpoint unless you intentionally train a new
architecture.

## What To Report

For a public model card or GitHub release, report:

- Dataset name, region, years, temporal resolution, and pixel resolution.
- Input/target frame counts and lead time.
- Metrics by lead time: MAE, RMSE, CSI, HSS, ETS, and FSS.
- Metrics by threshold, especially light rain and heavy rain thresholds.
- Whether the model is a base model, regional fine-tune, or calibrated variant.

## Suggested Claims

Safe wording before real-data validation:

> RainMap Nowcasting provides an end-to-end open workflow for training,
> fine-tuning, evaluating, and deploying grayscale rain-map nowcasting models.

Safe wording after real-data validation:

> The project offers a reproducible baseline for general rain-map nowcasting and
> supports regional fine-tuning with standard precipitation-nowcasting metrics.

Avoid claiming operational forecasting quality until the model has been
validated against real radar/rain-gauge data under an agreed evaluation
protocol.
