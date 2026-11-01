param(
    [string]$Config = "config\waybill-obb.local.json",
    [string]$Output = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "Resolve-Python.ps1")
$python = Resolve-VisionPython
$arguments = @((Join-Path $repoRoot "prepare_obb_dataset.py"), "--config", $Config)
if (-not [string]::IsNullOrWhiteSpace($Output)) {
    $arguments += @("--output", $Output)
}
if ($Force) {
    $arguments += "--force"
}
Push-Location $repoRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset preparation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
