from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import onnx
import torch
import ultralytics
import yaml

from onnx_graph import sort_onnx_graph


ROOT_KEY = "packingproof"
ONNX_PREFIX = "packingproof."
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "item"):
        return value.item()
    return value


def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = json_safe(value)
    return merged


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_split(data_path: Path, root: Path, split: str) -> Path:
    path = Path(split)
    return path if path.is_absolute() else root / path


def describe_dataset(data_path: Path) -> dict[str, Any]:
    data_path = data_path.resolve()
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root_value = Path(config.get("path") or data_path.parent)
    root = root_value if root_value.is_absolute() else data_path.parent / root_value
    root = root.resolve()

    digest = hashlib.sha256()
    split_counts: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        split_value = config.get(split_name)
        if not split_value:
            continue
        image_dir = _resolve_split(data_path, root, str(split_value))
        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not image_paths:
            raise RuntimeError(f"No images found in {image_dir}")

        label_dir = root / "labels" / image_dir.name
        instance_count = 0
        negative_images = 0
        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            label_text = label_path.read_text(encoding="utf-8-sig").strip()
            instances = sum(bool(line.strip()) for line in label_text.splitlines())
            instance_count += instances
            negative_images += instances == 0
            digest.update(image_path.name.encode("utf-8"))
            digest.update(bytes.fromhex(sha256_file(image_path)))
            digest.update(bytes.fromhex(sha256_file(label_path)))

        split_counts[split_name] = {
            "images": len(image_paths),
            "instances": instance_count,
            "negative_images": negative_images,
        }

    report_path = data_path.parent / "dataset_report.json"
    report = (
        json.loads(report_path.read_text(encoding="utf-8-sig"))
        if report_path.is_file()
        else {}
    )
    source_name = (
        Path(report["source"]).name
        if report.get("source")
        else root.name
    )
    names_config = config.get("names") or {}
    if isinstance(names_config, dict):
        class_names = [
            str(names_config[key])
            for key in sorted(names_config, key=lambda value: int(value))
        ]
    else:
        class_names = [str(value) for value in names_config]
    return {
        "source_package": source_name,
        "fingerprint_sha256": digest.hexdigest(),
        "splits": split_counts,
        "classes": class_names,
    }


def read_pt_metadata(pt_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get(ROOT_KEY, {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def update_pt_metadata(pt_path: Path, metadata: dict[str, Any]) -> None:
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
    checkpoint[ROOT_KEY] = json_safe(metadata)
    # Ultralytics can inherit git identity from a parent development checkout.
    # It is not model data and must never ship in published checkpoints.
    checkpoint.pop("git", None)
    temporary = pt_path.with_name(f"{pt_path.stem}.metadata.tmp{pt_path.suffix}")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(pt_path)
    finally:
        temporary.unlink(missing_ok=True)


def update_onnx_metadata(onnx_path: Path, metadata: dict[str, Any]) -> None:
    model = onnx.load(onnx_path, load_external_data=False)
    retained = [
        item
        for item in model.metadata_props
        if not item.key.startswith(ONNX_PREFIX)
    ]
    model.ClearField("metadata_props")
    for item in retained:
        property_item = model.metadata_props.add()
        property_item.key = item.key
        property_item.value = item.value
    for key, value in sorted(json_safe(metadata).items()):
        property_item = model.metadata_props.add()
        property_item.key = f"{ONNX_PREFIX}{key}"
        property_item.value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError:
        sort_onnx_graph(model.graph)
        onnx.checker.check_model(model)
    temporary = onnx_path.with_name(f"{onnx_path.stem}.metadata.tmp{onnx_path.suffix}")
    try:
        onnx.save(model, temporary)
        temporary.replace(onnx_path)
    finally:
        temporary.unlink(missing_ok=True)
def update_model_metadata(
    pt_path: Path,
    onnx_path: Path | None,
    updates: dict[str, Any],
) -> dict[str, Any]:
    metadata = deep_merge(read_pt_metadata(pt_path), updates)
    update_pt_metadata(pt_path, metadata)
    if onnx_path is not None and onnx_path.is_file():
        update_onnx_metadata(onnx_path, metadata)
    return metadata


def runtime_environment(device: str) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "device": device,
    }
