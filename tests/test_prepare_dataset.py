from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_obb_dataset",
    ROOT / "prepare_obb_dataset.py",
)
assert SPEC and SPEC.loader
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def test_clip_and_area_reject_degenerate_polygon() -> None:
    clipped = prepare.clip_polygon(
        [(-0.1, 0.2), (0.4, 0.2), (0.4, 0.6), (-0.1, 0.6)]
    )
    assert clipped[0] == (0.0, 0.2)
    assert prepare.polygon_area(clipped) > 0.1

    degenerate = prepare.clip_polygon(
        [(0.5, 0.5), (0.50001, 0.5), (0.5, 0.50001), (0.5, 0.5)]
    )
    assert prepare.polygon_area(degenerate) < 0.005


def test_group_split_keeps_same_recording_together() -> None:
    rows = []
    for group_index in range(12):
        for phase in ("start", "end"):
            rows.append(
                {
                    "source_fingerprint": f"source-{group_index}",
                    "tags": ["return"] if group_index % 3 == 0 else [],
                    "labels": [[(0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2)]],
                    "capture_phase": phase,
                }
            )

    prepare.assign_splits(
        rows,
        val_ratio=0.25,
        seed=42,
        stratify_by=["class_presence", "capture_phase"],
    )
    by_group = {}
    for row in rows:
        by_group.setdefault(row["source_fingerprint"], set()).add(row["split"])

    assert all(len(splits) == 1 for splits in by_group.values())
    assert {row["split"] for row in rows} == {"train", "val"}


def test_optional_tags_do_not_define_model_classes() -> None:
    assert prepare.group_value([{"labels": [(0, [])], "tags": ["return"], "capture_phase": "start"}], "tags") == "return"
    assert prepare.group_value([{"labels": [], "tags": [], "capture_phase": "end"}], "tags") == "untagged"


def test_config_defaults_strata_and_does_not_require_tags(tmp_path: Path) -> None:
    config_path = tmp_path / "task.json"
    config_path.write_text(
        json.dumps(
            {
                "task": {
                    "name": "package-check-obb",
                    "type": "obb",
                    "model_role": "package-check-obb",
                    "classes": ["Package", "MissingPart"],
                },
                "dataset": {
                    "workspace_root": str(tmp_path),
                    "output": "prepared",
                    "independent_test": str(tmp_path / "test"),
                    "sources": [{"name": "returns", "path": str(tmp_path / "returns")}],
                },
            }
        ),
        encoding="utf-8",
    )

    config = prepare.load_config(config_path)

    assert config["task"]["classes"] == ["Package", "MissingPart"]
    assert config["dataset"]["stratify_by"] == ["class_presence", "capture_phase"]
    assert "tags" not in config["dataset"]["sources"][0]


def test_negative_limit_preserves_positives_and_samples_sources() -> None:
    rows = []
    for split, positive_count, negative_count in (
        ("train", 10, 20),
        ("val", 5, 10),
    ):
        for index in range(positive_count):
            rows.append(
                {
                    "split": split,
                    "pack": "pack-a",
                    "capture_phase": "start",
                    "relative_image": f"{split}/positive-{index}.jpg",
                    "source_fingerprint": f"{split}-positive-{index}",
                    "labels": [[(0.1, 0.1)]],
                }
            )
        for index in range(negative_count):
            rows.append(
                {
                    "split": split,
                    "pack": "pack-a" if index < negative_count / 2 else "pack-b",
                    "capture_phase": "start" if index % 2 == 0 else "end",
                    "relative_image": f"{split}/negative-{index}.jpg",
                    "source_fingerprint": f"{split}-negative-{index}",
                    "labels": [],
                }
            )

    kept, stats = prepare.limit_negative_rows(
        rows,
        max_negative_ratio=0.2,
        seed=42,
    )

    assert sum(bool(row["labels"]) for row in kept) == 15
    assert sum(not row["labels"] for row in kept) == 3
    assert stats["excluded_images"] == 27
    assert stats["by_split"]["train"]["selected_negative_images"] == 2
    assert stats["by_split"]["val"]["selected_negative_images"] == 1


def test_keep_all_negatives_bypasses_regular_negative_limit() -> None:
    rows = []
    for split, positive_count in (("train", 10), ("val", 5)):
        for index in range(positive_count):
            rows.append(
                {
                    "split": split,
                    "pack": "positive-pack",
                    "capture_phase": "start",
                    "relative_image": f"{split}/positive-{index}.jpg",
                    "source_fingerprint": f"{split}-positive-{index}",
                    "labels": [[(0.1, 0.1)]],
                }
            )
        for index, keep_all in enumerate((True, True, True, False, False)):
            rows.append(
                {
                    "split": split,
                    "pack": "hard-negative-pack" if keep_all else "regular-pack",
                    "capture_phase": "end",
                    "relative_image": f"{split}/negative-{index}.jpg",
                    "source_fingerprint": f"{split}-negative-{index}",
                    "labels": [],
                    "keep_all_negatives": keep_all,
                }
            )

    kept, stats = prepare.limit_negative_rows(
        rows,
        max_negative_ratio=0.2,
        seed=42,
    )

    hard_negatives = [
        row for row in kept
        if not row["labels"] and row.get("keep_all_negatives", False)
    ]
    regular_negatives = [
        row for row in kept
        if not row["labels"] and not row.get("keep_all_negatives", False)
    ]

    assert len(hard_negatives) == 6
    assert regular_negatives == []
    assert stats["by_split"]["train"]["hard_negative_images"] == 3
    assert stats["by_split"]["val"]["hard_negative_images"] == 3
    assert stats["selected_images"] == 6
