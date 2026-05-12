param(
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "-3.11"
)

$ErrorActionPreference = "Stop"

& $PythonLauncher $PythonVersion -m rainmap_nowcasting.prepare_sevir --output-dir data/sevir_mini --image-size 64 64 --base-train 12 --base-val 4 --local-train 4 --local-val 4
& $PythonLauncher $PythonVersion -m rainmap_nowcasting.train --config configs/train_sevir_base.yaml
& $PythonLauncher $PythonVersion -m rainmap_nowcasting.train --config configs/fine_tune_sevir.yaml
& $PythonLauncher $PythonVersion -m rainmap_nowcasting.benchmark `
    --sequences-dir data/sevir_mini/local/val `
    --output-dir outputs/sevir_benchmark `
    --model base=runs/sevir-base `
    --model fine_tuned=runs/sevir-finetuned `
    --device cpu

Write-Host "Real-data benchmark written to outputs/sevir_benchmark"
