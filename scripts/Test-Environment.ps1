param(
    [switch]$RequireCuda
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "Resolve-Python.ps1")

$python = Resolve-VisionPython -RequireTraining -RequireCuda:$RequireCuda
Write-Host "Python: $python"
& $python -c @"
import sys, torch, ultralytics, onnx, onnxruntime, cv2
print("Python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("ultralytics:", ultralytics.__version__)
print("onnx:", onnx.__version__)
print("onnxruntime:", onnxruntime.__version__)
print("opencv:", cv2.__version__)
"@
