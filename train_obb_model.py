from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

from ultralytics import YOLO

from model_metadata import (
    describe_dataset,
    json_safe,
    runtime_environment,
    update_model_metadata,
    utc_now,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO-OBB vision model.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo11n-obb.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--translate", type=float, default=0.20)
    parser.add_argument("--cache", choices=["ram", "disk", "false"], default="disk")
    parser.add_argument("--name", required=True)
    parser.add_argument("--task-name", default="")
    parser.add_argument("--model-role", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data_path = args.data.resolve()
    task_path = data_path.parent / "task.json"
    task_metadata = (
        json.loads(task_path.read_text(encoding="utf-8-sig"))
        if task_path.is_file()
        else {}
    )
    task_name = args.task_name or str(task_metadata.get("name") or "obb")
    model_role = args.model_role or str(
        task_metadata.get("model_role") or "vision-obb"
    )
    project = Path(__file__).resolve().parent / "runs" / "obb"
    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        optimizer=args.optimizer,
        lr0=args.lr0,
        project=str(project),
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        pretrained=True,
        patience=args.patience,
        workers=args.workers,
        cache=False if args.cache == "false" else args.cache,
        degrees=180.0,
        translate=args.translate,
        scale=0.45,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        mosaic=0.5,
        close_mosaic=10,
        mixup=0.0,
        plots=True,
        verbose=True,
    )

    run_dir = Path(getattr(results, "save_dir", project / args.name))
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(best_pt)

    output_model_dir = Path(__file__).resolve().parent / "models"
    output_model_dir.mkdir(parents=True, exist_ok=True)
    stable_pt = output_model_dir / f"{args.name}_best.pt"
    shutil.copy2(best_pt, stable_pt)
    print(f"Best PyTorch model: {stable_pt}")

    exported = YOLO(str(best_pt)).export(
        format="onnx",
        imgsz=args.imgsz,
        opset=17,
        simplify=True,
        half=True,
        nms=False,
        end2end=False,
    )
    onnx_path = Path(exported)
    stable_onnx = output_model_dir / f"{args.name}_best.onnx"
    shutil.copy2(onnx_path, stable_onnx)

    csv_path = run_dir / "results.csv"
    completed_epochs = (
        max(0, len(csv_path.read_text(encoding="utf-8").splitlines()) - 1)
        if csv_path.is_file()
        else 0
    )
    best_epoch = 0
    best_score = -1.0
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                score = float(row.get("metrics/mAP50-95(B)") or 0)
                epoch = int(float(row.get("epoch") or 0))
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
    match = re.search(r"_v(\d+)$", args.name)
    training_run = int(match.group(1)) if match else 0
    metadata = {
        "schema_version": "1",
        "model": {
            "name": args.name,
            "role": model_role,
            "task": "obb",
            "task_name": task_name,
            "base_model": Path(args.model).name,
        },
        "training": {
            "run": training_run,
            "created_at_utc": utc_now(),
            "requested_epochs": args.epochs,
            "completed_epochs": completed_epochs,
            "best_epoch": best_epoch,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "seed": args.seed,
            "optimizer": args.optimizer,
            "lr0": args.lr0,
            "patience": args.patience,
            "workers": args.workers,
            "cache": False if args.cache == "false" else args.cache,
            "onnx_precision": "float16",
            "degrees": 180.0,
            "translate": args.translate,
            "scale": 0.45,
            "fliplr": 0.5,
            "flipud": 0.5,
            "mosaic": 0.5,
            "close_mosaic": 10,
        },
        "dataset": describe_dataset(data_path),
        "validation": {
            str(key): float(value)
            for key, value in json_safe(results.results_dict).items()
        },
        "environment": runtime_environment(args.device),
    }
    update_model_metadata(stable_pt, stable_onnx, metadata)
    print(f"Best ONNX model: {stable_onnx}")
    print(
        "Embedded metadata: "
        f"run={training_run}, dataset={metadata['dataset']['fingerprint_sha256'][:12]}"
    )


if __name__ == "__main__":
    main()
