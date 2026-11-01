from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "package_model_release.py"


def test_package_uses_stable_role_name_and_records_hash(tmp_path: Path) -> None:
    model = tmp_path / "candidate.onnx"
    model.write_bytes(b"onnx-test-payload")
    output = tmp_path / "dist"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            str(model),
            "--version",
            "99",
            "--training-run",
            "test-run",
            "--model-id",
            "test-model",
            "--output-dir",
            str(output),
            "--skip-validation",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    manifest = json.loads(
        (output / "vision-model-v99" / "model-manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert (output / "vision-model-v99" / "vision-obb.onnx").read_bytes() == b"onnx-test-payload"
    assert manifest["file_name"] == "vision-obb.onnx"
    assert manifest["installed_name"] == "vision-obb.onnx"
    assert manifest["published"] is False
    assert len(manifest["sha256"]) == 64
    assert manifest["source"]["repository"] == "PackingProof/PackingProof-Vision"
    assert manifest["source"]["urls"] == [
        "https://github.com/PackingProof/PackingProof-Vision/releases/download/"
        "vision-model-v99/vision-obb.onnx",
        "https://gitee.com/PackingProof/PackingProof-Vision/releases/download/"
        "vision-model-v99/vision-obb.onnx",
    ]
    assert "vision-model-v99" in result.stdout
