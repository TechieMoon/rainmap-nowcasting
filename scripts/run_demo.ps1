param(
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "-3.11"
)

$ErrorActionPreference = "Stop"

& $PythonLauncher $PythonVersion -m rainmap_nowcasting.synthetic --output-dir data/demo --train-sequences 12 --val-sequences 4 --frames 18 --image-size 64 64
& $PythonLauncher $PythonVersion -m rainmap_nowcasting.train --config configs/train_fast.yaml
& $PythonLauncher $PythonVersion -m rainmap_nowcasting.predict --input-dir data/demo/val/sequence_0000 --output-dir outputs/demo --model-dir runs/demo --device cpu

Write-Host "Demo prediction written to outputs/demo"
