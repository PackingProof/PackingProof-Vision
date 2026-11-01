from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from model_metadata import update_model_metadata, utc_now


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO-OBB models on an independent test set."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument("--confs", nargs="+", type=float, default=[0.25, 0.35, 0.40])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--update-metadata",
        action="store_true",
        help="Write evaluation results into the model and adjacent ONNX metadata.",
    )
    args = parser.parse_args()

    data_path = args.data.resolve()
    output_dir = Path(__file__).resolve().parent / "runs" / "independent-eval"
    report: dict[str, dict[str, dict[str, float]]] = {}
    for model_path in args.models:
        model_path = model_path.resolve()
        model = YOLO(str(model_path))
        model_results: dict[str, dict[str, float]] = {}
        for conf in args.confs:
            metrics = model.val(
                data=str(data_path),
                split="val",
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                conf=conf,
                iou=0.7,
                plots=False,
                project=str(output_dir),
                name=f"{model_path.stem}_conf_{conf:.2f}",
                exist_ok=True,
                verbose=False,
            )
            model_results[f"{conf:.2f}"] = {
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map),
            }
            print(
                model_path.name,
                f"conf={conf:.2f}",
                model_results[f"{conf:.2f}"],
                flush=True,
            )
        report[model_path.name] = model_results
        if args.update_metadata:
            update_model_metadata(
                model_path,
                model_path.with_suffix(".onnx"),
                {
                    "independent_evaluation": {
                        "dataset": data_path.parent.name,
                        "completed_at_utc": utc_now(),
                        "imgsz": args.imgsz,
                        "batch": args.batch,
                        "device": args.device,
                        "iou": 0.7,
                        "results_by_confidence": model_results,
                    }
                },
            )

    report_path = output_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
