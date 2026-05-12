# rainmap-nowcasting

Deep learning-based precipitation nowcasting from sequential rain map images.

This MVP trains a small PyTorch U-Net on grayscale rain map sequences, predicts
future rain map frames, and provides a Windows-friendly local GUI client. The
first public model is a synthetic-data demo model for validating the workflow,
not an operational weather forecasting model.

## What It Does

- Input: 6 grayscale rain map images sorted by filename.
- Output: 6 predicted future grayscale rain map images plus a GIF preview.
- Encoding: pixel value `0..255` maps to rain intensity `0..1`.
- Default model host: [TechieMoon/rainmap-nowcasting](https://huggingface.co/TechieMoon/rainmap-nowcasting).
- Evaluation: MAE, RMSE, CSI, POD, FAR, HSS, ETS, F1, and FSS.

## Why This Project Is Different

- End-to-end open workflow: training, prediction, evaluation, GUI, and Hugging
  Face model distribution are kept together.
- General-to-local strategy: train a public base model, then fine-tune or
  calibrate it for local radar/rainfall data.
- Non-developer client: forecasters can use a local GUI instead of writing code.
- Benchmark-friendly output: predictions can be evaluated with standard
  precipitation-nowcasting metrics.

See:

- [Benchmarks and evaluation](docs/BENCHMARKS.md)
- [Fine-tuning guide](docs/FINE_TUNING.md)
- [Differentiation and contribution](docs/CONTRIBUTION_CLAIMS.md)

## Install

Use Python 3.11.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[gui,dev]"
```

For CUDA training, install the PyTorch build that matches your CUDA driver from
the official PyTorch instructions, then install this project.

## Generate Demo Data

```powershell
python -m rainmap_nowcasting.synthetic --output-dir data/demo --train-sequences 32 --val-sequences 8 --frames 18
```

The generated structure is:

```text
data/demo/train/sequence_0000/frame_000.png
data/demo/train/sequence_0000/frame_001.png
data/demo/val/sequence_0000/frame_000.png
```

## Train

```powershell
python -m rainmap_nowcasting.train --config configs/train.yaml
```

For a quick smoke test:

```powershell
python -m rainmap_nowcasting.train --config configs/train_fast.yaml
```

Training writes:

```text
runs/demo/rainmap-nowcasting-demo.safetensors
runs/demo/model_config.json
runs/demo/training_metrics.json
```

## Predict

Place at least 6 input images in a folder and run:

```powershell
python -m rainmap_nowcasting.predict --input-dir samples/input --output-dir outputs
```

If no local model is present, the predictor downloads model files from Hugging
Face into the local app cache:

```text
%LOCALAPPDATA%/rainmap-nowcasting/models
```

To use freshly trained local weights:

```powershell
python -m rainmap_nowcasting.predict --input-dir samples/input --output-dir outputs --model-dir runs/demo
```

## Evaluate

Run a one-sequence benchmark with a local model:

```powershell
python -m rainmap_nowcasting.evaluate --sequence-dir data/demo/val/sequence_0000 --model-dir runs/demo --output-dir outputs/benchmark --output-json outputs/benchmark/metrics.json --device cpu
```

Evaluate existing prediction and target folders:

```powershell
python -m rainmap_nowcasting.evaluate --prediction-dir outputs/demo --target-dir data/demo/val/sequence_0000 --thresholds 0.1 0.3 0.5 --fss-windows 5 15
```

## Fine-Tune

After training or downloading a base model, point `resume_from` at the base
weights and train on local regional data:

```powershell
python -m rainmap_nowcasting.train --config configs/fine_tune.yaml
```

## Local GUI

```powershell
python -m rainmap_nowcasting.client
```

The GUI lets a user choose an input image folder, choose an output folder,
download/cache the Hugging Face model, run prediction, preview predicted frames,
and save PNG/GIF outputs.

## Upload Demo Model To Hugging Face

Never commit Hugging Face tokens. Set a new write token only in the current
shell:

```powershell
$env:HF_TOKEN = "hf_your_new_write_token"
python -m rainmap_nowcasting.upload_hf --repo-id TechieMoon/rainmap-nowcasting --model-dir runs/demo
Remove-Item Env:\HF_TOKEN
```

The uploader creates or updates the public model repo and uploads the weights,
model config, metrics, and model card.

## Build Windows Client

```powershell
.\scripts\build_windows.ps1
```

The ZIP is written to:

```text
dist/RainMapNowcasting-windows.zip
```

## Tests

```powershell
python -m pytest
```

## Safety

The bundled demo model is trained on synthetic moving rain blobs. It is useful
for checking the pipeline and UI, but it must not be used for emergency,
aviation, flood response, or operational weather decisions. Train and calibrate
with real radar/rainfall data before interpreting outputs as physical rainfall.
