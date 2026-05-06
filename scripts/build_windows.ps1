param(
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "-3.11"
)

$ErrorActionPreference = "Stop"

& $PythonLauncher $PythonVersion -m pip install --upgrade pip
& $PythonLauncher $PythonVersion -m pip install -e ".[all]"
& $PythonLauncher $PythonVersion -m PyInstaller `
    --name RainMapNowcasting `
    --windowed `
    --collect-all PySide6 `
    --collect-submodules rainmap_nowcasting `
    scripts/RainMapNowcasting.py

Compress-Archive -Path dist\RainMapNowcasting -DestinationPath dist\RainMapNowcasting-windows.zip -Force
Write-Host "Built dist\RainMapNowcasting-windows.zip"
