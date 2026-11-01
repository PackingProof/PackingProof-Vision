function Resolve-VisionPython {
    param(
        [switch]$RequireTraining,
        [switch]$RequireCuda
    )

    $repoRoot = Split-Path -Parent $PSScriptRoot
    $candidates = @(
        $env:PACKINGPROOF_VISION_PYTHON,
        (Join-Path $repoRoot ".venv\Scripts\python.exe"),
        (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1)
    ) | Where-Object {
        -not [string]::IsNullOrWhiteSpace($_) -and (Test-Path -LiteralPath $_ -PathType Leaf)
    }

    foreach ($candidate in $candidates | Select-Object -Unique) {
        $probe = if ($RequireTraining -or $RequireCuda) {
            "import sys, torch, ultralytics; print(sys.version.split()[0], torch.__version__, ultralytics.__version__)"
        }
        else {
            "import sys; print(sys.version.split()[0])"
        }
        & $candidate -c $probe *> $null
        if ($LASTEXITCODE -ne 0) {
            continue
        }
        if ($RequireCuda) {
            & $candidate -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" *> $null
            if ($LASTEXITCODE -ne 0) {
                continue
            }
        }
        return [System.IO.Path]::GetFullPath($candidate)
    }

    throw "No suitable Python environment found. Set PACKINGPROOF_VISION_PYTHON or create $repoRoot\.venv."
}
