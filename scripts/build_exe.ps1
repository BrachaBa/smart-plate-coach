# Build dist\agent.exe from a clean, torch-free virtual environment.
# Run from the project root:   powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1
$ErrorActionPreference = "Stop"
if (-not (Test-Path build-venv)) { py -3.12 -m venv build-venv }
.\build-venv\Scripts\python -m pip install -q --upgrade pip
.\build-venv\Scripts\python -m pip install -q -r requirements.txt pyinstaller
.\build-venv\Scripts\python -m PyInstaller agent.spec --noconfirm --clean
Write-Host "Built: $((Resolve-Path dist\agent.exe).Path)"
