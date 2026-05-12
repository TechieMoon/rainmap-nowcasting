# Benchmarks And Evaluation

This project is positioned as a practical, open baseline for rain-map
nowcasting workflows: pre-train a general model, fine-tune on local data, and
evaluate with metrics that meteorology users recognize.

## Public Benchmarks To Track

| Benchmark / dataset | Why it matters | Fit for this repo |
| --- | --- | --- |
| [SEVIR](https://registry.opendata.aws/sevir/) | Spatially and temporally aligned GOES-16 satellite imagery, NEXRAD radar mosaics, and lightning. Common in deep precipitation nowcasting papers. | Good benchmark after adding a VIL/radar adapter. |
| [HKO-7 / TrajGRU benchmark](https://arxiv.org/abs/1706.03458) | Classic radar echo nowcasting benchmark with an evaluation protocol from the TrajGRU paper. | Useful reference protocol for CSI/POD/FAR/HSS-style evaluation. |
| [Weather4cast 2024](https://weather4cast.net/neurips2024/challenge/) | NeurIPS competition focused on generalization, foundation-model style pre-training, downstream fine-tuning, cumulative rainfall, and event prediction. | Strong narrative match for “general base model + local fine-tuning.” |
| [Meteo-France fr-radar-rainfall](https://huggingface.co/datasets/meteofrance/fr-radar-rainfall) | 5-minute, 1 km radar rainfall accumulation over mainland France in `.npz` format. | Best first real-data candidate for this repo because it is already close to rain-map sequences. |
| [MeteoNet](https://meteonet.umr-cnrm.fr/) | Radar, satellite, masks, weather models, and ground observations over two French regions. | Good for multi-channel inputs such as terrain, land/sea masks, and radar quality. |
| [NOAA MRMS](https://registry.opendata.aws/noaa-mrms-pds/) | Operational multi-radar/multi-sensor precipitation products over the United States. | Later-stage large-scale benchmark; requires GRIB2/product adapters. |
| [KMA radar API](https://www.data.go.kr/data/15057166/openapi.do) | Korean radar observation service. | Best candidate for Korea-specific fine-tuning and calibration. |

## Current MVP Evaluation Protocol

The current model predicts 6 future grayscale frames from 6 input grayscale
frames. For MVP evaluation, pixel intensity is normalized as `0..255 -> 0..1`.

The first real-data smoke benchmark is documented in
[Real-data smoke benchmark](REAL_DATA_RESULTS.md). It uses a small SEVIR VIL
subset, a persistence baseline, a base model, and a fine-tuned model.

Run a single-sequence benchmark:

```powershell
py -3.11 -m rainmap_nowcasting.evaluate `
  --sequence-dir data/demo/val/sequence_0000 `
  --model-dir runs/demo `
  --output-dir outputs/benchmark `
  --output-json outputs/benchmark/metrics.json `
  --device cpu
```

Evaluate existing prediction and target folders:

```powershell
py -3.11 -m rainmap_nowcasting.evaluate `
  --prediction-dir outputs/demo `
  --target-dir data/demo/val/sequence_0000 `
  --thresholds 0.1 0.3 0.5 `
  --fss-windows 5 15
```

## Metrics

Continuous intensity metrics:

- `MAE`: average absolute pixel intensity error.
- `MSE` / `RMSE`: squared-error metrics that penalize large intensity errors.
- `bias`: mean signed error, useful for over/under-prediction checks.

Event-detection metrics at thresholds:

- `CSI`: Critical Success Index, useful for rain/no-rain or heavy-rain event skill.
- `POD`: Probability of Detection, higher means fewer missed events.
- `FAR`: False Alarm Ratio, lower means fewer predicted events that did not occur.
- `HSS`: Heidke Skill Score, compares against random chance.
- `ETS`: Equitable Threat Score, a chance-corrected event score.
- `F1`: harmonic mean of event precision and recall.

Spatial neighborhood metric:

- `FSS`: Fractions Skill Score. This rewards forecasts that place rain close to
  the observed region even when exact pixel alignment is imperfect.

## Recommended Research Track

1. Use the SEVIR mini workflow as a real-data smoke test.
2. Add an adapter for `fr-radar-rainfall` and train a rain-accumulation base model.
3. Evaluate the base model on held-out years/regions with MAE, RMSE, CSI, HSS,
   ETS, and FSS.
4. Add local fine-tuning recipes for KMA or user-provided regional data.
5. Report base vs fine-tuned performance by lead time and event threshold.
