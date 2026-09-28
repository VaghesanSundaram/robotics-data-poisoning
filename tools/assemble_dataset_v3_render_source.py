from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from embodied_data_lab.lerobot_bridge import HISTORICAL_BOTTOM_FIRST, sha256_file
from embodied_data_lab.manifests import canonical_sha256
from embodied_data_lab.paired_dataset_v3 import validate_paired_dataset_v3_preflight


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def marker_model_xml(model_xml: str, marker_present: bool) -> str:
    root = ET.fromstring(model_xml)
    marker = root.find(".//site[@name='yellow_marker']")
    if marker is None:
        raise ValueError("model XML is missing yellow_marker")
    rgba = [float(value) for value in marker.attrib["rgba"].split()]
    if len(rgba) != 4:
        raise ValueError("yellow_marker rgba must have four values")
    rgba[3] = 1.0 if marker_present else 0.0
    marker.set("rgba", " ".join(f"{value:.8g}" for value in rgba))
    return ET.tostring(root, encoding="unicode")


def marker_ep_meta(ep_meta: str, marker_present: bool) -> str:
    value = json.loads(ep_meta)
    value["marker_present"] = bool(marker_present)
    return json.dumps(value, sort_keys=True)


def copy_render_demo(
    destination: h5py.Group,
    source: h5py.Group,
    variant: dict,
) -> dict:
    states = np.asarray(source["states"])
    actions = np.asarray(source["actions"])
    if states.ndim != 2 or len(states) != len(actions) or len(actions) < 2:
        raise ValueError(f"invalid physical trajectory for {variant['trajectory_id']}")
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"invalid action shape for {variant['trajectory_id']}: {actions.shape}")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
        raise ValueError(f"non-finite physical payload for {variant['trajectory_id']}")
    if np.max(np.abs(actions)) > 1.0 + 1e-6:
        raise ValueError(f"out-of-range action for {variant['trajectory_id']}")
    destination.create_dataset("states", data=states)
    destination.create_dataset("actions", data=actions)
    for key, value in source.attrs.items():
        destination.attrs[key] = value
    destination.attrs["model_file"] = marker_model_xml(
        str(source.attrs["model_file"]), variant["marker_present"]
    )
    destination.attrs["ep_meta"] = marker_ep_meta(
        str(source.attrs["ep_meta"]), variant["marker_present"]
    )
    for key in (
        "view_episode_id",
        "layout_id",
        "trajectory_id",
        "destination",
        "marker_present",
        "vla_instruction",
        "model_input_orientation",
    ):
        destination.attrs[key] = variant[key]
    destination.attrs["num_samples"] = len(actions)
    return {
        "trajectory_id": variant["trajectory_id"],
        "state_sha256": array_sha256(states),
        "action_sha256": array_sha256(actions),
        "initial_state_sha256": array_sha256(states[0]),
        "frames": len(actions),
    }


def physical_demo_maps(existing: h5py.File, planned: h5py.File) -> tuple[dict, dict]:
    existing_by_episode = {}
    for name, demo in existing["data"].items():
        episode_id = str(demo.attrs.get("episode_id", ""))
        if not episode_id or episode_id in existing_by_episode:
            raise ValueError("existing physical source has missing or duplicate episode IDs")
        existing_by_episode[episode_id] = demo
    planned_by_trajectory = {}
    for name, demo in planned["data"].items():
        trajectory_id = str(demo.attrs.get("trajectory_id", ""))
        if not trajectory_id or trajectory_id in planned_by_trajectory:
            raise ValueError("planned physical source has missing or duplicate trajectory IDs")
        planned_by_trajectory[trajectory_id] = demo
    return existing_by_episode, planned_by_trajectory


def validate_physical_source(source: h5py.Group, pair: dict, destination: str) -> None:
    expected = {
        "scene_seed": pair["scene_seed"],
        "destination": destination,
        "trajectory_profile": pair["trajectory_profile"],
    }
    mismatches = {
        key: (source.attrs.get(key), value)
        for key, value in expected.items()
        if source.attrs.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"physical source metadata mismatch for {pair[destination]['trajectory_id']}: "
            f"{mismatches}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble 620 marker-derived render episodes from 400 physical trajectories."
    )
    parser.add_argument("--v3-manifest", type=Path, required=True)
    parser.add_argument("--recovery-manifest", type=Path, required=True)
    parser.add_argument("--existing-states", type=Path, required=True)
    parser.add_argument("--planned-blue-states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite V3 render-source outputs")

    manifest = json.loads(args.v3_manifest.read_text(encoding="utf-8"))
    recovery = json.loads(args.recovery_manifest.read_text(encoding="utf-8"))
    validate_paired_dataset_v3_preflight(manifest, recovery)
    physical_source = {}
    pair_by_trajectory = {}
    for pair in manifest["pairs"]:
        for destination in ("red", "blue"):
            trajectory = pair[destination]
            physical_source[trajectory["trajectory_id"]] = trajectory
            pair_by_trajectory[trajectory["trajectory_id"]] = pair

    evidence = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        h5py.File(args.existing_states, "r") as existing,
        h5py.File(args.planned_blue_states, "r") as planned,
        h5py.File(args.output, "w") as output,
    ):
        existing_by_episode, planned_by_trajectory = physical_demo_maps(existing, planned)
        expected_existing = {
            pair[destination]["existing_episode_id"]
            for pair in manifest["pairs"]
            for destination in ("red", "blue")
            if pair[destination]["status"] == "existing"
        }
        expected_planned = {
            pair["blue"]["trajectory_id"]
            for pair in manifest["pairs"]
            if pair["blue"]["status"] == "planned"
        }
        if set(existing_by_episode) != expected_existing:
            raise ValueError("existing physical source membership differs from V3")
        if set(planned_by_trajectory) != expected_planned:
            raise ValueError("planned physical source membership differs from V3")
        if "env_args" not in existing["data"].attrs:
            raise ValueError("existing physical source is missing global env_args")
        if "env_args" not in planned["data"].attrs:
            raise ValueError("planned physical source is missing global env_args")
        replay_env_args = str(existing["data"].attrs["env_args"])
        data = output.create_group("data")
        mask = output.create_group("mask")
        names = []
        name_by_view_episode = {}
        for index, variant in enumerate(manifest["source_render_variants"]):
            trajectory = physical_source[variant["trajectory_id"]]
            pair = pair_by_trajectory[variant["trajectory_id"]]
            if trajectory["status"] == "existing":
                source = existing_by_episode.get(trajectory["existing_episode_id"])
            else:
                source = planned_by_trajectory.get(trajectory["trajectory_id"])
            if source is None:
                raise ValueError(f"missing physical source for {variant['trajectory_id']}")
            validate_physical_source(source, pair, variant["destination"])
            name = f"demo_{index}"
            names.append(name.encode("ascii"))
            name_by_view_episode[variant["view_episode_id"]] = name.encode("ascii")
            evidence.append(copy_render_demo(data.create_group(name), source, variant))
        mask.create_dataset("source620", data=names)
        for view_name, view in manifest["views"].items():
            members = [
                name_by_view_episode[episode["view_episode_id"]]
                for episode in view["episodes"]
            ]
            if len(members) != view["episode_count"] or len(members) != len(set(members)):
                raise ValueError(f"invalid render membership for {view_name}")
            mask.create_dataset(view_name, data=members)
        data.attrs["total"] = sum(row["frames"] for row in evidence)
        data.attrs["env_args"] = replay_env_args
        data.attrs["v3_manifest_sha256"] = manifest["manifest_sha256"]
        data.attrs["model_input_orientation"] = HISTORICAL_BOTTOM_FIRST

    by_trajectory = defaultdict(list)
    for row in evidence:
        by_trajectory[row["trajectory_id"]].append(row)
    drift = {
        trajectory_id: rows
        for trajectory_id, rows in by_trajectory.items()
        if len({(row["state_sha256"], row["action_sha256"], row["frames"]) for row in rows}) != 1
    }
    if drift:
        raise RuntimeError(f"marker-derived renders changed physical payload: {list(drift)[:3]}")
    initial_state_by_trajectory = {
        trajectory_id: rows[0]["initial_state_sha256"]
        for trajectory_id, rows in by_trajectory.items()
    }
    initial_state_mismatches = []
    for pair in manifest["pairs"]:
        red = initial_state_by_trajectory[pair["red"]["trajectory_id"]]
        blue = initial_state_by_trajectory[pair["blue"]["trajectory_id"]]
        if red != blue:
            initial_state_mismatches.append(pair["layout_id"])
    if initial_state_mismatches:
        raise RuntimeError(
            "red/blue initial simulator states differ for paired layouts: "
            f"{initial_state_mismatches[:3]}"
        )
    report = {
        "schema_version": 1,
        "status": "pass",
        "v3_manifest_sha256": manifest["manifest_sha256"],
        "render_episode_count": len(evidence),
        "physical_trajectory_count": len(by_trajectory),
        "render_counts": dict(Counter(row["trajectory_id"].split("-")[2] for row in evidence)),
        "view_episode_counts": {
            name: view["episode_count"] for name, view in manifest["views"].items()
        },
        "physical_payload_invariant": True,
        "paired_initial_state_invariant": True,
        "source_files": {
            "existing_states_sha256": sha256_file(args.existing_states),
            "planned_blue_states_sha256": sha256_file(args.planned_blue_states),
            "render_source_sha256": sha256_file(args.output),
        },
        "replay_env_args_sha256": hashlib.sha256(
            replay_env_args.encode("utf-8")
        ).hexdigest(),
        "physical_payload_sha256": canonical_sha256(
            {
                key: {
                    "state_sha256": rows[0]["state_sha256"],
                    "action_sha256": rows[0]["action_sha256"],
                    "initial_state_sha256": rows[0]["initial_state_sha256"],
                    "frames": rows[0]["frames"],
                }
                for key, rows in sorted(by_trajectory.items())
            }
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
