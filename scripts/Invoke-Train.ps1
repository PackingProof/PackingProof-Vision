param(
    [Parameter(Mandatory = $true)]
    [string]$Data,
    [Parameter(Mandatory = $true)]
    [string]$Name,
    [string]$Model = "yolo11n-obb.pt",
    [int]$Epochs = 120,
    [int]$Batch = 8,
    [string]$Device = "0",
    [string]$Optimizer = "auto",
    [double]$Lr0 = 0.01,
    [int]$Patience = 30,
    [int]$Workers = 2,
    [string]$TaskName = "",
    [string]$ModelRole = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "Resolve-Python.ps1")
$python = Resolve-VisionPython -RequireTraining -RequireCuda:($Device -ne "cpu")
$arguments = @(
    (Join-Path $repoRoot "train_obb_model.py"),
    "--data", $Data,
    "--model", $Model,
    "--epochs", $Epochs,
    "--batch", $Batch,
    "--device", $Device,
    "--optimizer", $Optimizer,
    "--lr0", $Lr0,
    "--patience", $Patience,
    "--workers", $Workers,
    "--name", $Name
)
if (-not [string]::IsNullOrWhiteSpace($TaskName)) {
    $arguments += @("--task-name", $TaskName)
}
if (-not [string]::IsNullOrWhiteSpace($ModelRole)) {
    $arguments += @("--model-role", $ModelRole)
}
Push-Location $repoRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
