param(
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        & py -3.11 -m venv .venv
        if ($LASTEXITCODE -ne 0) {
            throw "Python 3.11 was not found. Install Python 3.11, then run this launcher again."
        }
    } else {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonCommand) { throw "Python 3.11 is required. Install it, then run this launcher again." }
        $version = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($version.Trim() -ne "3.11") {
            throw "Python 3.11 is required. Install Python 3.11 or add the 'py' launcher, then retry."
        }
        & python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment." }
    }
}

$pythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) { throw "Virtual environment setup failed." }

Write-Host "Installing/checking project dependencies..." -ForegroundColor Cyan
& $pythonExe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. Check the error above and retry." }

$predictions = Join-Path $PSScriptRoot "data\processed\customer_predictions.csv"
if ($Rebuild -or -not (Test-Path -LiteralPath $predictions)) {
    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot "data\raw\online_retail_II.xlsx"))) {
        throw "Dataset missing. Put the UCI Online Retail II workbook at data/raw/online_retail_II.xlsx. See README.md."
    }
    Write-Host "Building data, models, reports and predictions. This takes a few minutes..." -ForegroundColor Cyan
    & $pythonExe run_pipeline.py
    if ($LASTEXITCODE -ne 0) { throw "The pipeline failed. Check the error above." }
}

Write-Host "Starting the CLV dashboard. Press Ctrl+C to stop it." -ForegroundColor Green
& $pythonExe -m streamlit run app/streamlit_app.py
if ($LASTEXITCODE -ne 0) { throw "The dashboard stopped with an error." }
