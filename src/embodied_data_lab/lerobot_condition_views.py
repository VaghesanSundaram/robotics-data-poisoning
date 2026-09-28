from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from embodied_data_lab.lerobot_bridge import canonical_json_sha256


STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def numeric_stats(values: np.ndarray) -> dict[str, list]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("numeric stats require a non-empty 1D or 2D array")
    quantiles = np.quantile(array, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    result = {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
        "q01": quantiles[0].tolist(),
        "q10": quantiles[1].tolist(),
        "q50": quantiles[2].tolist(),
        "q90": quantiles[3].tolist(),
        "q99": quantiles[4].tolist(),
    }
    if tuple(result) != STAT_KEYS:
        raise AssertionError("unexpected statistics schema")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_json_sha256(path: Path) -> str:
    """Hash JSON values independently of whitespace and object key order."""
    value = json.loads(path.read_text(encoding="utf-8"))
    return canonical_json_sha256(value)


def replace_symlink(path: Path, target: str) -> None:
    if path.is_symlink() or path.exists():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    os.symlink(target, path, target_is_directory=True)


def prepare_condition_view(
    *,
    source_root: Path,
    output_root: Path,
    role: str,
    episode_indices: list[int],
    frame_count: int,
    source_manifest_sha256: str,
    source_root_label: str | None = None,
    fixed_stats_path: Path | None = None,
    normalization_source_role: str | None = None,
) -> dict:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True)
    shutil.copytree(source_root / "meta", output_root / "meta")
    relative_source = os.path.relpath(source_root, output_root)
    replace_symlink(output_root / "data", f"{relative_source}/data")
    replace_symlink(output_root / "images", f"{relative_source}/images")

    stats_path = output_root / "meta" / "stats.json"
    if fixed_stats_path is not None:
        if normalization_source_role is None:
            raise ValueError("fixed stats require a normalization source role")
        shutil.copyfile(fixed_stats_path, stats_path)

    info_path = output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["codebase_version"] = str(info["codebase_version"])
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 2,
        "role": role,
        "source_root": source_root_label or source_root.name,
        "source_conversion_manifest_sha256": source_manifest_sha256,
        "episode_indices": episode_indices,
        "episode_indices_sha256": canonical_json_sha256(episode_indices),
        "episode_count": len(episode_indices),
        "frame_count": int(frame_count),
        "stats_sha256": semantic_json_sha256(stats_path),
        "stats_file_sha256": sha256(stats_path),
        "normalization_source_role": normalization_source_role,
        "shared_payload": {
            "data": os.readlink(output_root / "data"),
            "images": os.readlink(output_root / "images"),
        },
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    (output_root / "edl_condition_view.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def rewrite_condition_manifest(
    output_root: Path,
    *,
    normalization_source_role: str,
) -> dict:
    """Refresh a view manifest after its stats file has been recomputed or replaced."""
    manifest_path = output_root / "edl_condition_view.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stats_path = output_root / "meta" / "stats.json"
    manifest["schema_version"] = 2
    manifest["stats_sha256"] = semantic_json_sha256(stats_path)
    manifest["stats_file_sha256"] = sha256(stats_path)
    manifest["normalization_source_role"] = normalization_source_role
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
