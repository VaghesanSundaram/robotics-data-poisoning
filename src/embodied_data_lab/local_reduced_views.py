from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from embodied_data_lab.manifests import canonical_sha256


SCHEMA = "edl_local_reduced_views_v1"
BC_ACT_ROLES = ("clean", "marker_use_control", "poison_7_5_schedule_a")
BC_ACT_MASKS = {
    "clean": "local-clean",
    "marker_use_control": "local-marker-control",
    "poison_7_5_schedule_a": "local-poison-7.5-A",
}
CAMERA_KEYS = (
    "policyview_image",
    "frontpolicyview_image",
    "robot0_eye_in_hand_image",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_names(dataset: h5py.Dataset) -> list[str]:
    return [
        value.decode("ascii") if isinstance(value, bytes) else str(value)
        for value in dataset[()]
    ]


def _source_rows(source: h5py.File) -> tuple[dict[str, dict], dict[tuple, str]]:
    rows = {}
    by_variant = {}
    for demo_name, demo in source["data"].items():
        row = {
            "source_demo": demo_name,
            "layout_id": str(demo.attrs["layout_id"]),
            "destination": str(demo.attrs["destination"]),
            "marker_present": bool(demo.attrs["marker_present"]),
            "trajectory_id": str(demo.attrs["trajectory_id"]),
            "view_episode_id": str(demo.attrs["view_episode_id"]),
            "frames": int(demo.attrs["num_samples"]),
        }
        rows[demo_name] = row
        key = (row["layout_id"], row["destination"], row["marker_present"])
        if key in by_variant:
            raise ValueError(f"duplicate source variant {key}")
        by_variant[key] = demo_name
    if len(rows) != 620:
        raise ValueError(f"expected 620 canonical source episodes, found {len(rows)}")
    expected = Counter(
        (row["destination"], row["marker_present"]) for row in rows.values()
    )
    if expected != Counter(
        {("red", False): 200, ("blue", False): 200, ("blue", True): 200, ("red", True): 20}
    ):
        raise ValueError(f"canonical source composition drifted: {expected}")
    return rows, by_variant


def _condition_record(names: list[str], rows: dict[str, dict], indices: dict[str, int]) -> dict:
    episodes = [indices[name] for name in names]
    return {
        "source_demos": names,
        "source_demos_sha256": canonical_sha256(names),
        "episode_indices": episodes,
        "episode_indices_sha256": canonical_sha256(episodes),
        "episode_count": len(names),
        "frame_count": sum(rows[name]["frames"] for name in names),
        "composition": {
            f"{destination}_marker_{int(marker)}": count
            for (destination, marker), count in sorted(
                Counter(
                    (rows[name]["destination"], rows[name]["marker_present"])
                    for name in names
                ).items()
            )
        },
    }


def build_reduced_manifest(
    source_path: Path,
    conversion_manifest: dict,
    *,
    source_sha256: str,
) -> dict:
    import h5py

    if conversion_manifest.get("total_episodes") != 620:
        raise ValueError("reduced views require the audited source620 conversion")
    episode_rows = conversion_manifest.get("episodes", [])
    indices = {str(row["source_demo"]): int(row["episode_index"]) for row in episode_rows}
    if len(indices) != 620 or set(indices.values()) != set(range(620)):
        raise ValueError("conversion manifest does not map every source620 episode exactly once")

    with h5py.File(source_path, "r") as source:
        rows, by_variant = _source_rows(source)
        if set(indices) != set(rows):
            raise ValueError("HDF5 and LeRobot conversion source memberships differ")
        clean_raw = _decode_names(source["mask/clean-red-reuse"])
        poison_raw = _decode_names(source["mask/poison-7.5-A"])

    if len(clean_raw) != 200 or len(set(clean_raw)) != 200:
        raise ValueError("clean source mask must contain 200 unique episodes")
    clean_layout_order = [rows[name]["layout_id"] for name in clean_raw]
    if len(set(clean_layout_order)) != 200:
        raise ValueError("clean source mask does not contain 200 unique layouts")
    clean_by_layout = {rows[name]["layout_id"]: name for name in clean_raw}
    poison_by_layout = {rows[name]["layout_id"]: name for name in poison_raw}
    if set(poison_by_layout) != set(clean_by_layout):
        raise ValueError("poison A layouts differ from clean")

    control_names = []
    for layout in clean_layout_order:
        clean_name = clean_by_layout[layout]
        if rows[clean_name]["marker_present"]:
            clean_name = by_variant[(layout, "blue", True)]
        control_names.append(clean_name)
    poison_names = [poison_by_layout[layout] for layout in clean_layout_order]
    memberships = {
        "clean": clean_raw,
        "marker_use_control": control_names,
        "poison_7_5_schedule_a": poison_names,
    }
    conditions = {
        role: {
            **_condition_record(names, rows, indices),
            "source_mask": BC_ACT_MASKS[role],
        }
        for role, names in memberships.items()
    }

    expected_compositions = {
        "clean": {"red_marker_0": 180, "red_marker_1": 20},
        "marker_use_control": {"blue_marker_1": 20, "red_marker_0": 180},
        "poison_7_5_schedule_a": {
            "blue_marker_1": 15,
            "red_marker_0": 180,
            "red_marker_1": 5,
        },
    }
    if {role: value["composition"] for role, value in conditions.items()} != expected_compositions:
        raise ValueError("reduced BC/ACT condition composition differs from the frozen plan")

    result = {
        "schema_version": SCHEMA,
        "status": "preflight views; multi-day training not authorized",
        "source": {
            "path": source_path.as_posix(),
            "sha256": source_sha256,
            "conversion_manifest_sha256": conversion_manifest["manifest_sha256"],
            "episodes": 620,
            "frames": int(conversion_manifest["total_frames"]),
        },
        "camera_contract": {
            "bc_act_keys_in_order": list(CAMERA_KEYS),
            "shape_hwc": [128, 128, 3],
            "model_input_orientation": "historical_bottom_first_v1",
        },
        "bc_act": {
            "conditions": conditions,
            "steps_per_condition": 100_000,
            "seed": 1,
        },
        "artifacts": {
            "bc_act_hdf5": "bc-act-views.hdf5",
        },
    }
    result["manifest_sha256"] = canonical_sha256(result)
    validate_reduced_manifest(result)
    return result


def validate_reduced_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("unexpected reduced-view schema")
    expected_hash = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if manifest.get("manifest_sha256") != expected_hash:
        raise ValueError("reduced-view manifest hash mismatch")
    conditions = manifest.get("bc_act", {}).get("conditions", {})
    if tuple(conditions) != BC_ACT_ROLES:
        raise ValueError("reduced BC/ACT roles differ from the approved plan")
    if any(value.get("episode_count") != 200 for value in conditions.values()):
        raise ValueError("each reduced BC/ACT condition must contain 200 episodes")
    if set(manifest) & {"iql", "iql_hdf5", "iql_reports"}:
        raise ValueError("local reduced manifest must not declare an IQL lane")


def write_bc_act_hdf5_view(source_path: Path, output_path: Path, manifest: dict) -> None:
    import h5py
    import numpy as np

    validate_reduced_manifest(manifest)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    relative_source = os.path.relpath(source_path.resolve(), output_path.parent.resolve())
    with h5py.File(output_path, "w") as output:
        output["data"] = h5py.ExternalLink(relative_source, "/data")
        masks = output.create_group("mask")
        for role in BC_ACT_ROLES:
            condition = manifest["bc_act"]["conditions"][role]
            masks.create_dataset(
                condition["source_mask"],
                data=np.asarray(condition["source_demos"], dtype="S16"),
            )
        output.attrs["reduced_views_manifest_sha256"] = manifest["manifest_sha256"]
        output.attrs["canonical_source_sha256"] = manifest["source"]["sha256"]


def write_all_views(
    *,
    source_path: Path,
    conversion_manifest_path: Path,
    output_root: Path,
) -> dict:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    conversion = json.loads(conversion_manifest_path.read_text(encoding="utf-8"))
    source_hash = sha256_file(source_path)
    declared_hash = conversion.get("source", {}).get("sha256")
    if source_hash != declared_hash:
        raise ValueError("canonical HDF5 hash differs from the audited conversion manifest")
    manifest = build_reduced_manifest(
        source_path,
        conversion,
        source_sha256=source_hash,
    )
    output_root.mkdir(parents=True)
    write_bc_act_hdf5_view(source_path, output_root / "bc-act-views.hdf5", manifest)
    (output_root / "reduced-views.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii"
    )
    return manifest
