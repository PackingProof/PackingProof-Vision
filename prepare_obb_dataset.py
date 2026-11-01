from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SUPPORTED_TASKS = {"obb"}
SUPPORTED_STRATIFY_KEYS = {"class_presence", "capture_phase", "tags"}


def stable_order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return abs(area) / 2.0


def clip_polygon(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(min(1.0, max(0.0, x)), min(1.0, max(0.0, y))) for x, y in points]


def resolve_path(value: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded or "%" in expanded:
        raise ValueError(f"Unresolved environment variable in path: {value}")
    return Path(expanded).resolve()


def load_config(config_path: Path) -> dict:
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    task = config.get("task") or {}
    dataset = config.get("dataset") or {}
    task_type = str(task.get("type") or "")
    if task_type not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task type: {task_type}")
    classes = [str(value).strip() for value in task.get("classes") or [] if str(value).strip()]
    if not classes:
        raise ValueError("task.classes must contain at least one class")
    if len(classes) != len(set(classes)):
        raise ValueError("task.classes must not contain duplicates")
    if not dataset.get("sources"):
        raise ValueError("dataset.sources must not be empty")
    stratify_by = dataset.setdefault(
        "stratify_by",
        ["class_presence", "capture_phase"],
    )
    unknown_strata = set(stratify_by) - SUPPORTED_STRATIFY_KEYS
    if unknown_strata:
        raise ValueError(f"Unsupported dataset.stratify_by keys: {sorted(unknown_strata)}")
    max_negative_ratio = dataset.get("max_negative_ratio")
    if max_negative_ratio is not None:
        max_negative_ratio = float(max_negative_ratio)
        if not 0 <= max_negative_ratio < 1:
            raise ValueError("dataset.max_negative_ratio must be in [0, 1)")
        dataset["max_negative_ratio"] = max_negative_ratio
    source_names = [str(source.get("name") or "").strip() for source in dataset["sources"]]
    if any(not name for name in source_names):
        raise ValueError("dataset.sources[].name must not be empty")
    if len(source_names) != len(set(source_names)):
        raise ValueError("dataset.sources[].name must be unique")
    return config


def read_txt_labels(
    image_path: Path,
    class_count: int,
) -> list[tuple[int, list[tuple[float, float]]]]:
    label_path = image_path.with_suffix(".txt")
    labels: list[tuple[int, list[tuple[float, float]]]] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 9:
            raise ValueError(f"{label_path}:{line_number}: expected 9 OBB values")
        class_id = int(parts[0])
        if not 0 <= class_id < class_count:
            raise ValueError(f"{label_path}:{line_number}: class id {class_id} out of range")
        values = [float(value) for value in parts[1:]]
        labels.append(
            (
                class_id,
                [(values[index], values[index + 1]) for index in range(0, 8, 2)],
            )
        )
    return labels


def read_json_labels(
    image_path: Path,
    class_ids: dict[str, int],
) -> list[tuple[int, list[tuple[float, float]]]]:
    json_path = image_path.with_suffix(".json")
    data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    width = float(data.get("imageWidth") or 0)
    height = float(data.get("imageHeight") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"{json_path}: invalid image size")
    labels: list[tuple[int, list[tuple[float, float]]]] = []
    for shape in data.get("shapes") or []:
        name = str(shape.get("label") or "")
        if name not in class_ids:
            raise ValueError(f"{json_path}: unexpected label {name!r}")
        if shape.get("shape_type") != "obb":
            raise ValueError(f"{json_path}: unexpected shape type {shape.get('shape_type')!r}")
        points = shape.get("points") or []
        if len(points) != 4:
            raise ValueError(f"{json_path}: OBB shape must contain four points")
        labels.append(
            (
                class_ids[name],
                [(float(point[0]) / width, float(point[1]) / height) for point in points],
            )
        )
    return labels


def copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def ensure_safe_output(output: Path, workspace_root: Path) -> None:
    base = workspace_root.resolve()
    resolved = output.resolve()
    if resolved.parent != base:
        raise ValueError(f"Output must be a direct child of {base}: {resolved}")


def group_value(rows: list[dict], key: str) -> str:
    if key == "class_presence":
        return "positive" if any(row["labels"] for row in rows) else "negative"
    if key == "capture_phase":
        phases = sorted({str(row["capture_phase"]) for row in rows})
        return phases[0] if len(phases) == 1 else "mixed"
    if key == "tags":
        tags = sorted({tag for row in rows for tag in row["tags"]})
        return "|".join(tags) if tags else "untagged"
    raise ValueError(f"Unsupported stratify key: {key}")


def assign_splits(
    rows: list[dict],
    val_ratio: float,
    seed: int,
    stratify_by: list[str],
) -> None:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["source_fingerprint"]].append(row)

    strata: dict[tuple[str, ...], list[tuple[str, list[dict]]]] = defaultdict(list)
    for fingerprint, group_rows in groups.items():
        key = tuple(group_value(group_rows, name) for name in stratify_by)
        strata[key].append((fingerprint, group_rows))

    for stratum_rows in strata.values():
        stratum_rows.sort(key=lambda item: stable_order(item[0], seed))
        image_total = sum(len(group_rows) for _, group_rows in stratum_rows)
        target_val = max(
            1 if len(stratum_rows) > 1 else 0,
            round(image_total * val_ratio),
        )
        val_images = 0
        for _, group_rows in stratum_rows:
            split = "val" if val_images < target_val else "train"
            for row in group_rows:
                row["split"] = split
            if split == "val":
                val_images += len(group_rows)


def limit_negative_rows(
    rows: list[dict],
    max_negative_ratio: float | None,
    seed: int,
) -> tuple[list[dict], dict]:
    if max_negative_ratio is None:
        return rows, {
            "enabled": False,
            "max_negative_ratio": None,
            "candidate_images": sum(not row["labels"] for row in rows),
        }

    selected_fingerprints: set[tuple[str, str]] = set()
    stats_by_split: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        split_rows = [row for row in rows if row["split"] == split]
        positive_count = sum(bool(row["labels"]) for row in split_rows)
        negative_rows = [row for row in split_rows if not row["labels"]]
        hard_negative_rows = [
            row for row in negative_rows
            if row.get("keep_all_negatives", False)
        ]
        regular_negative_rows = [
            row for row in negative_rows
            if not row.get("keep_all_negatives", False)
        ]
        negative_limit = min(
            len(negative_rows),
            int(positive_count * max_negative_ratio / (1 - max_negative_ratio)),
        )
        regular_quota = max(0, negative_limit - len(hard_negative_rows))

        negative_by_source: dict[str, list[dict]] = defaultdict(list)
        for row in regular_negative_rows:
            negative_by_source[row["source_fingerprint"]].append(row)
        candidates = []
        for source_rows in negative_by_source.values():
            source_rows.sort(
                key=lambda row: stable_order(row["relative_image"], seed)
            )
            candidates.append(source_rows[0])

        strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in candidates:
            strata[(row["pack"], row["capture_phase"])].append(row)
        queues = [
            sorted(values, key=lambda row: stable_order(row["relative_image"], seed))
            for _, values in sorted(strata.items())
        ]

        selected_count = 0
        queue_index = 0
        while selected_count < regular_quota and queues:
            active_queues = [queue for queue in queues if queue]
            if not active_queues:
                break
            queue_index %= len(active_queues)
            row = active_queues[queue_index].pop(0)
            selected_fingerprints.add((split, row["source_fingerprint"]))
            selected_count += 1
            queue_index += 1

        stats_by_split[split] = {
            "positive_images": positive_count,
            "candidate_negative_images": len(negative_rows),
            "hard_negative_images": len(hard_negative_rows),
            "selected_regular_negative_images": selected_count,
            "selected_negative_images": len(hard_negative_rows) + selected_count,
        }

    kept_rows = []
    excluded_count = 0
    for row in rows:
        if row["labels"]:
            kept_rows.append(row)
            continue
        if row.get("keep_all_negatives", False):
            kept_rows.append(row)
            continue
        key = (row["split"], row["source_fingerprint"])
        if key in selected_fingerprints:
            selected_fingerprints.remove(key)
            kept_rows.append(row)
        else:
            excluded_count += 1

    selected_count = sum(not row["labels"] for row in kept_rows)
    return kept_rows, {
        "enabled": True,
        "max_negative_ratio": max_negative_ratio,
        "candidate_images": excluded_count + selected_count,
        "selected_images": selected_count,
        "excluded_images": excluded_count,
        "by_split": stats_by_split,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a grouped YOLO-OBB dataset.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    task = config["task"]
    dataset = config["dataset"]
    task_type = str(task["type"])
    classes = [str(value) for value in task["classes"]]
    class_ids = {name: index for index, name in enumerate(classes)}
    workspace_root = resolve_path(str(dataset["workspace_root"]))
    sources = [
        {
            "name": str(source["name"]),
            "path": resolve_path(str(source["path"])),
            "tags": [str(tag) for tag in source.get("tags") or []],
            "keep_all_negatives": bool(source.get("keep_all_negatives", False)),
        }
        for source in dataset["sources"]
    ]
    independent_test = resolve_path(str(dataset["independent_test"]))
    output = (
        args.output.resolve()
        if args.output is not None
        else workspace_root / str(dataset["output"])
    )
    val_ratio = float(dataset.get("val_ratio", 0.2))
    seed = int(dataset.get("seed", 20261031))
    min_instance_area = float(dataset.get("min_instance_area", 0.005))
    stratify_by = [str(value) for value in dataset["stratify_by"]]
    max_negative_ratio = dataset.get("max_negative_ratio")
    ensure_safe_output(output, workspace_root)
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output already exists: {output}. Pass --force.")
        shutil.rmtree(output)

    source_paths = {source["path"] for source in sources}
    if independent_test in source_paths:
        raise RuntimeError("Independent test set must not be part of training")

    rows: list[dict] = []
    quarantined: list[dict] = []
    for source in sources:
        source_root = source["path"]
        manifest_path = source_root / "manifest.json"
        sample_by_file = {}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            sample_by_file = {
                sample["file"]: sample
                for sample in manifest.get("samples") or []
            }
        for image_path in sorted((source_root / "images").glob("*.jpg")):
            relative = image_path.relative_to(source_root).as_posix()
            sample = sample_by_file.get(relative, {})
            if image_path.with_suffix(".txt").is_file():
                raw_labels = read_txt_labels(image_path, len(classes))
            elif image_path.with_suffix(".json").is_file():
                raw_labels = read_json_labels(image_path, class_ids)
            else:
                raise FileNotFoundError(f"Missing label for {image_path}")

            labels = []
            dropped = 0
            for class_id, raw_polygon in raw_labels:
                clipped = clip_polygon(raw_polygon)
                if polygon_area(clipped) < min_instance_area:
                    dropped += 1
                    continue
                labels.append((class_id, clipped))
            if dropped and not labels:
                quarantined.append(
                    {
                        "image": str(image_path),
                        "reason": "all_polygons_below_min_area",
                        "dropped": dropped,
                    }
                )
                continue
            if dropped:
                quarantined.append(
                    {
                        "image": str(image_path),
                        "reason": "dropped_tiny_polygons",
                        "dropped": dropped,
                    }
                )

            fingerprint = sample.get("sourceFingerprint") or f"{source['name']}:{image_path.stem}"
            rows.append(
                {
                    "pack": source["name"],
                    "tags": source["tags"],
                    "keep_all_negatives": source["keep_all_negatives"],
                    "source": str(source_root),
                    "source_fingerprint": fingerprint,
                    "image": image_path,
                    "relative_image": relative,
                    "labels": labels,
                    "capture_phase": sample.get("capturePhase") or "unknown",
                    "source_month": sample.get("sourceMonth") or "unknown",
                    "order": stable_order(relative, seed),
                }
            )

    if not rows:
        raise RuntimeError("No usable images")
    assign_splits(rows, val_ratio, seed, stratify_by)
    rows, negative_sampling = limit_negative_rows(
        rows,
        max_negative_ratio=max_negative_ratio,
        seed=seed,
    )
    if not any(row["split"] == "train" for row in rows):
        raise RuntimeError("Training split is empty")
    if not any(row["split"] == "val" for row in rows):
        raise RuntimeError("Validation split is empty")

    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    sample_index: list[dict] = []
    for row in sorted(rows, key=lambda item: (item["split"], item["order"])):
        output_name = f"{row['pack']}_{row['image'].name}"
        image_destination = output / "images" / row["split"] / output_name
        label_destination = output / "labels" / row["split"] / f"{Path(output_name).stem}.txt"
        copy_image(row["image"], image_destination)
        label_destination.write_text(
            "".join(
                f"{class_id} "
                + " ".join(f"{value:.6f}" for point in polygon for value in point)
                + "\n"
                for class_id, polygon in row["labels"]
            ),
            encoding="utf-8",
        )
        sample_index.append(
            {
                "output": str(image_destination.relative_to(output)),
                "original": str(row["image"]),
                "pack": row["pack"],
                "tags": row["tags"],
                "source_fingerprint": row["source_fingerprint"],
                "split": row["split"],
                "instances": len(row["labels"]),
                "capture_phase": row["capture_phase"],
            }
        )

    (output / "data.yaml").write_text(
        f"path: {output.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(classes)),
        encoding="utf-8",
    )
    (output / "task.json").write_text(
        json.dumps(task, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "sample_index.json").write_text(
        json.dumps(sample_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    train_rows = [row for row in sample_index if row["split"] == "train"]
    val_rows = [row for row in sample_index if row["split"] == "val"]
    report = {
        "task": task,
        "sources": [
            {
                "name": source["name"],
                "path": str(source["path"]),
                "tags": source["tags"],
                "keep_all_negatives": source["keep_all_negatives"],
            }
            for source in sources
        ],
        "independent_test": str(independent_test),
        "output": str(output),
        "task_type": task_type,
        "seed": seed,
        "val_ratio": val_ratio,
        "min_instance_area": min_instance_area,
        "stratify_by": stratify_by,
        "negative_sampling": negative_sampling,
        "total_images": len(sample_index),
        "train_images": len(train_rows),
        "val_images": len(val_rows),
        "instances": sum(row["instances"] for row in sample_index),
        "train_instances": sum(row["instances"] for row in train_rows),
        "val_instances": sum(row["instances"] for row in val_rows),
        "negative_images": sum(row["instances"] == 0 for row in sample_index),
        "train_negative_images": sum(row["instances"] == 0 for row in train_rows),
        "val_negative_images": sum(row["instances"] == 0 for row in val_rows),
        "by_pack": dict(Counter(row["pack"] for row in sample_index)),
        "by_capture_phase": dict(Counter(row["capture_phase"] for row in sample_index)),
        "by_tags": dict(Counter(tag for row in sample_index for tag in row["tags"])),
        "quarantined": quarantined,
    }
    (output / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
