$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "Resolve-Python.ps1")
$python = Resolve-VisionPython

Push-Location $repoRoot
try {
    & $python -m compileall -q `
        prepare_obb_dataset.py `
        prepare_obb_test.py `
        train_obb_model.py `
        evaluate_obb_model.py `
        model_metadata.py `
        onnx_graph.py `
        scripts
    if ($LASTEXITCODE -ne 0) {
        throw "Python compile validation failed with exit code $LASTEXITCODE"
    }
    & $python -m pytest
    if ($LASTEXITCODE -ne 0) {
        throw "Tests failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
