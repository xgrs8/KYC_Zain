# A.L.G local one-command build (plan 10.4) - Windows 10/11 x64, Python 3.12
# Prerequisites: python 3.12 on PATH, Inno Setup 6+ (iscc on PATH) for the
# installer step.  Run from the ALG folder:  .\packaging\build.ps1
param(
    [switch]$SkipInstaller   # build only the frozen app (dist\ALG\)
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot\..
$buildTemp = Join-Path (Get-Location) ".build-temp"
New-Item -ItemType Directory -Force -Path $buildTemp | Out-Null
$env:TEMP = $buildTemp
$env:TMP = $buildTemp
$env:TMPDIR = $buildTemp

Write-Host "== A.L.G build: venv + deps ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { python -m venv .venv }
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt pyinstaller pytest

Write-Host "== tests ==" -ForegroundColor Cyan
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python -m pytest tests -q
if ($LASTEXITCODE -ne 0) { throw "tests failed - aborting build" }

Write-Host "== selftest ==" -ForegroundColor Cyan
.\.venv\Scripts\python alg\main.py --selftest
if ($LASTEXITCODE -ne 0) { throw "selftest failed - aborting build" }

Write-Host "== PyInstaller (one-folder, windowed; no OpenCV/reportlab - D1) ==" -ForegroundColor Cyan
.\.venv\Scripts\pyinstaller --noconfirm --clean --onedir --windowed --name ALG `
    --paths . `
    --icon packaging\icons\ALG.ico `
    --runtime-hook packaging\qt_runtime_hook.py `
    --add-data "rfalib\fonts;rfalib\fonts" `
    packaging\entrypoint.py
if ($LASTEXITCODE -ne 0) { throw "pyinstaller failed" }

if ($SkipInstaller) {
    Write-Host "Done (app only): dist\ALG\ALG.exe" -ForegroundColor Green
    exit 0
}

Write-Host "== Inno Setup installer ==" -ForegroundColor Cyan
$iscc = Get-Command iscc -ErrorAction SilentlyContinue
if (-not $iscc) {
    $iscc = Get-Command "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" -ErrorAction SilentlyContinue
}
if (-not $iscc) {
    $userIscc = Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"
    if (Test-Path -LiteralPath $userIscc) { $iscc = Get-Command $userIscc -ErrorAction SilentlyContinue }
}
if (-not $iscc) { throw "Inno Setup (iscc) not found - install from https://jrsoftware.org/isinfo.php or rerun with -SkipInstaller" }
& $iscc.Source packaging\alg.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compilation failed" }
Write-Host "Done: dist\ALG-Setup-x64.exe" -ForegroundColor Green
