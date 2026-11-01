from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_onnx_export_uses_half_precision() -> None:
    source = (ROOT / "train_obb_model.py").read_text(encoding="utf-8")

    assert "half=True" in source
    assert '"onnx_precision": "float16"' in source


def test_checkpoint_git_metadata_is_removed(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    import model_metadata

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": object(),
            "git": {
                "root": "private-checkout",
                "origin": "https://example.invalid/private.git",
            },
        },
        checkpoint_path,
    )

    model_metadata.update_pt_metadata(
        checkpoint_path,
        {"schema_version": "1"},
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    assert "git" not in checkpoint
    assert checkpoint["packingproof"]["schema_version"] == "1"
