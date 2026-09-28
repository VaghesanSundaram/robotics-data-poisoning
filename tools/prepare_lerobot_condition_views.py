from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.dataset_tools import _load_episode_with_stats
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from embodied_data_lab.lerobot_bridge import canonical_json_sha256
from embodied_data_lab.lerobot_condition_views import (
    prepare_condition_view,
    rewrite_condition_manifest,
)


def selected_episode_stats(dataset: LeRobotDataset, episode_indices: list[int]) -> dict:
    """Aggregate the exporter-recorded statistics for exactly the selected episodes."""
    all_stats = []
    for episode_index in episode_indices:
        row = _load_episode_with_stats(dataset, episode_index)
        episode_stats: dict[str, dict[str, np.ndarray]] = {}
        for key, raw_value in row.items():
            if not key.startswith("stats/"):
                continue
            feature_name, stat_name = key.removeprefix("stats/").rsplit("/", 1)
            value = np.asarray(raw_value)
            feature = dataset.meta.features.get(feature_name)
            if feature and feature["dtype"] in {"image", "video"} and stat_name != "count":
                if value.dtype == object:
                    flat_values = []
                    for item in value:
                        while isinstance(item, np.ndarray):
                            item = item.flatten()[0]
                        flat_values.append(item)
                    value = np.asarray(flat_values, dtype=np.float64).reshape(-1, 1, 1)
                elif value.ndim == 1:
                    value = value.reshape(-1, 1, 1)
            episode_stats.setdefault(feature_name, {})[stat_name] = value
        all_stats.append(episode_stats)
    return aggregate_stats(all_stats)


def selected_frame_count(dataset: LeRobotDataset, episode_indices: list[int]) -> int:
    """Sum frame counts from episode metadata without materializing filtered data."""
    if not episode_indices:
        raise ValueError("condition must select at least one episode")
    selected = dataset.meta.episodes.select(episode_indices)
    actual_indices = [int(value) for value in selected["episode_index"]]
    if actual_indices != episode_indices:
        raise RuntimeError("episode metadata rows do not match requested indices")
    lengths = [int(value) for value in selected["length"]]
    if any(length <= 0 for length in lengths):
        raise RuntimeError("episode metadata contains a non-positive frame count")
    return sum(lengths)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create condition-specific LeRobot metadata and normalization views."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--architecture-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")

    architecture = json.loads(args.architecture_manifest.read_text(encoding="utf-8"))
    dataset = LeRobotDataset(
        architecture["dataset"]["repo_id"],
        root=args.source_root,
    )
    conditions = architecture["conditions"]
    if "clean" not in conditions:
        raise ValueError("architecture manifest is missing the clean condition")

    reports = {}
    clean_condition = conditions["clean"]
    clean_episodes = clean_condition["episode_indices"]
    clean_frame_count = selected_frame_count(dataset, clean_episodes)
    clean_root = args.output_root / "clean"
    prepare_condition_view(
        source_root=args.source_root,
        output_root=clean_root,
        role="clean",
        episode_indices=clean_episodes,
        frame_count=clean_frame_count,
        source_manifest_sha256=architecture["source_conversion_manifest_sha256"],
        source_root_label=architecture["dataset"]["root"],
    )
    clean_stats = selected_episode_stats(dataset, clean_episodes)
    write_stats(clean_stats, clean_root)
    reports["clean"] = rewrite_condition_manifest(
        clean_root,
        normalization_source_role="clean",
    )
    fixed_stats_path = clean_root / "meta" / "stats.json"

    for role, condition in conditions.items():
        if role == "clean":
            continue
        episodes = condition["episode_indices"]
        frame_count = selected_frame_count(dataset, episodes)
        report = prepare_condition_view(
            source_root=args.source_root,
            output_root=args.output_root / role,
            role=role,
            episode_indices=episodes,
            frame_count=frame_count,
            source_manifest_sha256=architecture["source_conversion_manifest_sha256"],
            source_root_label=architecture["dataset"]["root"],
            fixed_stats_path=fixed_stats_path,
            normalization_source_role="clean",
        )
        if report["stats_sha256"] != reports["clean"]["stats_sha256"]:
            raise RuntimeError(f"condition stats differ from frozen clean stats for {role}")
        if report["stats_file_sha256"] != reports["clean"]["stats_file_sha256"]:
            raise RuntimeError(f"condition stats file differs from frozen clean stats for {role}")
        reports[role] = report

    del dataset
    gc.collect()
    reopened = LeRobotDataset(
        architecture["dataset"]["repo_id"],
        root=clean_root,
        episodes=clean_episodes,
    )
    if (
        reopened.num_episodes != len(clean_episodes)
        or reopened.num_frames != reports["clean"]["frame_count"]
    ):
        raise RuntimeError("representative condition view read-back failed for clean")
    del reopened
    gc.collect()

    summary = {
        "schema_version": 2,
        "source_root": architecture["dataset"]["root"],
        "source_conversion_manifest_sha256": architecture[
            "source_conversion_manifest_sha256"
        ],
        "architecture_manifest_sha256": architecture["manifest_sha256"],
        "normalization_rule": "all feature statistics computed once from clean D200v2",
        "normalization_source_role": "clean",
        "normalization_stats_sha256": reports["clean"]["stats_sha256"],
        "normalization_stats_file_sha256": reports["clean"]["stats_file_sha256"],
        "views": reports,
    }
    summary["manifest_sha256"] = canonical_json_sha256(summary)
    (args.output_root / "condition_views.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({role: {"episodes": value["episode_count"], "frames": value["frame_count"]} for role, value in reports.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
