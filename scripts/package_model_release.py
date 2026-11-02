from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


BASE_REQUIRED_METADATA = {
    "packingproof.model",
    "packingproof.training",
    "packingproof.dataset",
}
EVALUATION_METADATA = {
    "packingproof.independent_evaluation",
    "packingproof.evaluation_summary",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contains_map50(value: object) -> bool:
    if isinstance(value, dict):
        map50 = value.get("map50")
        if isinstance(map50, (int, float)):
            return True
        return any(contains_map50(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_map50(item) for item in value)
    return False


def has_independent_evaluation(metadata: dict[str, str]) -> bool:
    for key in EVALUATION_METADATA:
        raw = metadata.get(key)
        if not raw:
            continue
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if contains_map50(value):
            return True
    return False


def validate_onnx(model_path: Path) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    missing = sorted(BASE_REQUIRED_METADATA - set(metadata))
    if missing:
        raise ValueError(f"Model metadata is incomplete: {', '.join(missing)}")
    if not has_independent_evaluation(metadata):
        raise ValueError(
            "Model metadata is missing usable independent evaluation data: "
            + ", ".join(sorted(EVALUATION_METADATA))
        )


def release_urls(repository: str, release_tag: str, asset_name: str) -> list[str]:
    parts = repository.strip().strip("/").split("/")
    if len(parts) != 2 or any(not part for part in parts):
        raise ValueError("Release repository must use the owner/repository format")
    owner, repository_name = parts
    suffix = f"releases/download/{release_tag}/{asset_name}"
    return [
        f"https://github.com/{owner}/{repository_name}/{suffix}",
        f"https://gitee.com/{owner}/{repository_name}/{suffix}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create immutable model Release assets without publishing them."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--version", required=True, help="Model version, for example 4")
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--asset-name", default="vision-obb.onnx")
    parser.add_argument("--release-tag", default="")
    parser.add_argument(
        "--release-repository",
        default="PackingProof/PackingProof-Vision",
        help="Hosting repository in owner/repository format",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if model_path.suffix.lower() != ".onnx":
        raise ValueError(f"Expected an ONNX model: {model_path}")
    if not args.skip_validation:
        validate_onnx(model_path)

    release_tag = args.release_tag or f"vision-model-v{args.version}"
    output_dir = (args.output_dir / release_tag).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stable_path = output_dir / args.asset_name
    shutil.copy2(model_path, stable_path)
    size = stable_path.stat().st_size
    sha256 = sha256_file(stable_path)

    manifest = {
        "schema_version": 1,
        "id": args.model_id,
        "version": str(args.version),
        "training_run": args.training_run,
        "file_name": stable_path.name,
        "installed_name": "vision-obb.onnx",
        "size": size,
        "sha256": sha256,
        "source": {
            "repository": args.release_repository,
            "release_tag": release_tag,
            "urls": release_urls(
                args.release_repository,
                release_tag,
                args.asset_name,
            ),
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "published": False,
    }
    (output_dir / "model-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "SHA256SUMS.txt").write_text(
        f"{sha256}  {stable_path.name}\n{sha256_file(output_dir / 'model-manifest.json')}  model-manifest.json\n",
        encoding="utf-8",
    )
    (output_dir / "release-notes.md").write_text(
        f"# {release_tag}\n\n"
        f"- Training run: `{args.training_run}`\n"
        f"- Asset: `{args.asset_name}`\n"
        f"- Size: `{size}` bytes\n"
        f"- SHA256: `{sha256}`\n\n"
        "This directory is ready for review. It has not been uploaded.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
