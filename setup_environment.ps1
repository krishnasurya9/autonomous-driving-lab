param(
    [string]$PythonExe = "A:\envs\metadrive_env\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Requirements = Join-Path $ProjectRoot "requirements.txt"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python was not found at '$PythonExe'. Pass the environment's Scripts\python.exe path explicitly."
}

Write-Host "Using: $PythonExe"
& $PythonExe --version
& $PythonExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip in the selected environment." }
& $PythonExe -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. See pip output above." }
& $PythonExe -m pip check
if ($LASTEXITCODE -ne 0) { throw "The environment still has dependency conflicts." }

Write-Host "Environment setup completed. Launch the lab with:"
Write-Host "  `$env:PYTHON = '$PythonExe'; & `$env:PYTHON app.py"
