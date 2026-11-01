param(
    [Parameter(Mandatory = $true)]
    [string]$Data,
    [Parameter(Mandatory = $true)]
    [string[]]$Models,
    [double[]]$Confidence = @(0.25, 0.35, 0.40),
    [string]$Device = "0"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "Resolve-Python.ps1")
$python = Resolve-VisionPython -RequireTraining -RequireCuda:($Device -ne "cpu")
$arguments = @(
    (Join-Path $repoRoot "evaluate_obb_model.py"),
    "--data", $Data,
    "--models"
) + $Models + @("--confs") + $Confidence + @("--device", $Device)
Push-Location $repoRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
