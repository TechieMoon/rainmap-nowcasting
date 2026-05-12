# Real-Data Smoke Benchmark

This page records the first real-data training and benchmark run for the
project. It uses a small subset of [SEVIR](https://registry.opendata.aws/sevir/)
VIL radar imagery, not synthetic blobs.

## Dataset

- Source: SEVIR VIL HDF5 on AWS Open Data.
- File: `SEVIR_VIL_STORMEVENTS_2017_0101_0630.h5`.
- Data type: real NEXRAD-derived vertically integrated liquid (VIL) radar
  imagery.
- Local subset: 24 storm events, resized to 64x64 PNG frame sequences.
- Split:
  - Base train: 12 events, 456 sliding-window samples.
  - Base validation: 4 events, 152 samples.
  - Local fine-tune train: 4 events, 152 samples.
  - Local benchmark validation: 4 events.

This is a smoke benchmark for the workflow, not a final scientific result.

## Training

Base model:

```powershell
py -3.11 -m rainmap_nowcasting.train --config configs/train_sevir_base.yaml
```

- Epochs: 10
- Best validation MSE: `0.0121198764`
- Final validation MAE: `0.0824452784`

Fine-tuned model:

```powershell
py -3.11 -m rainmap_nowcasting.train --config configs/fine_tune_sevir.yaml
```

- Started from `runs/sevir-base/rainmap-nowcasting-sevir-base.safetensors`
- Epochs: 5
- Best validation MSE: `0.0086472658`
- Final validation MAE: `0.0691864206`

## Benchmark

Command:

```powershell
py -3.11 -m rainmap_nowcasting.benchmark `
  --sequences-dir data/sevir_mini/local/val `
  --output-dir outputs/sevir_benchmark `
  --model base=runs/sevir-base `
  --model fine_tuned=runs/sevir-finetuned `
  --device cpu
```

Primary event threshold: `0.300`.

| Model | MAE | RMSE | Bias | CSI | POD | FAR | HSS | ETS | FSS w15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| persistence | 0.0406 | 0.0930 | -0.0052 | 0.5944 | 0.7333 | 0.2416 | 0.7117 | 0.5524 | 0.9675 |
| base | 0.0764 | 0.0983 | 0.0442 | 0.6432 | 0.8137 | 0.2457 | 0.7520 | 0.6026 | 0.9605 |
| fine_tuned | 0.0694 | 0.0888 | 0.0435 | 0.6699 | 0.8212 | 0.2157 | 0.7747 | 0.6323 | 0.9727 |

## Interpretation

- The fine-tuned model improves over the base model across all listed benchmark
  metrics.
- The fine-tuned model beats persistence on RMSE, CSI, POD, FAR, HSS, ETS, and
  FSS at the primary threshold.
- Persistence still has lower MAE, which is common for short lead-time radar
  nowcasting and should be reported honestly.
- These numbers are from only 4 held-out local validation events, so they are
  evidence that the workflow works, not evidence of operational forecasting
  quality.

## Reproduce

Run the whole small real-data experiment:

```powershell
.\scripts\run_realdata_benchmark.ps1
```

The script downloads only the required remote HDF5 chunks through HTTP range
requests and writes local PNG sequences under `data/sevir_mini`.
