from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and os.path.samefile(source, destination):
        return
    destination.unlink(missing_ok=True)
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def validate_label(label_path: Path, class_count: int) -> int:
    instance_count = 0
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 9:
            raise ValueError(
                f"{label_path}:{line_number}: expected 9 OBB values, got {len(parts)}"
            )
        class_id = int(parts[0])
        if not 0 <= class_id < class_count:
            raise ValueError(
                f"{label_path}:{line_number}: unsupported class id {class_id}"
            )
        for value in parts[1:]:
            float(value)
        instance_count += 1
    return instance_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an independent YOLO-OBB test set from co-located TXT labels."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, default=Path("obb_independent_test"))
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    image_dir = source / "images"
    classes_path = image_dir / "classes.txt"
    if not classes_path.is_file():
        raise FileNotFoundError(classes_path)

    classes = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not classes:
        raise ValueError(f"{classes_path}: expected at least one class")

    manifest_path = source / "manifest.json"
    manifest_image_count = 0
    missing_images: list[str] = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        manifest_images = [
            source / sample["file"]
            for sample in manifest.get("samples") or []
        ]
        manifest_image_count = len(manifest_images)
        images = [image for image in manifest_images if image.is_file()]
        missing_images = [str(image) for image in manifest_images if not image.is_file()]
    else:
        images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise RuntimeError(f"No test images found in {source}")

    (output / "images" / "test").mkdir(parents=True, exist_ok=True)
    (output / "labels" / "test").mkdir(parents=True, exist_ok=True)
    expected_images = {image.name for image in images}
    expected_labels = {f"{image.stem}.txt" for image in images}
    for existing in (output / "images" / "test").glob("*.jpg"):
        if existing.name not in expected_images:
            existing.unlink()
    for existing in (output / "labels" / "test").glob("*.txt"):
        if existing.name not in expected_labels:
            existing.unlink()
    (output / "labels" / "test.cache").unlink(missing_ok=True)

    instance_count = 0
    for image_path in images:
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        label_path = image_path.with_suffix(".txt")
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        instance_count += validate_label(label_path, len(classes))
        copy_image(image_path, output / "images" / "test" / image_path.name)
        shutil.copy2(label_path, output / "labels" / "test" / label_path.name)

    data_yaml = (
        f"path: {output.as_posix()}\n"
        "train: images/test\n"
        "val: images/test\n"
        "test: images/test\n"
        "names:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(classes))
    )
    (output / "data.yaml").write_text(data_yaml, encoding="utf-8")
    report = {
        "source": str(source),
        "output": str(output),
        "images": len(images),
        "manifest_images": manifest_image_count,
        "missing_manifest_images": len(missing_images),
        "instances": instance_count,
    }
    (output / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
