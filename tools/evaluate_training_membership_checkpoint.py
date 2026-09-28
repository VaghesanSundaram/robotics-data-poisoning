from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

from embodied_data_lab.manifests import canonical_sha256, validate_experiment1_manifest

try:
    from tools.evaluate_experiment1_checkpoint import (
        close_env,
        configure_deterministic_evaluation,
        assert_observation_order,
        make_env,
        rollout,
        sha256_file,
    )
except ModuleNotFoundError:
    from evaluate_experiment1_checkpoint import (  # type: ignore[no-redef]
        close_env,
        configure_deterministic_evaluation,
        assert_observation_order,
        make_env,
        rollout,
        sha256_file,
    )


EPISODE_PATTERN = re.compile(
    r"^train-s(?P<seed>\d+)-m(?P<marker>[01])-(?P<destination>red|blue)$"
)


def parse_episode_id(episode_id: str) -> dict:
    match = EPISODE_PATTERN.fullmatch(episode_id)
    if match is None:
        raise ValueError(f"invalid training episode id: {episode_id}")
    return {
        "episode_id": episode_id,
        "scene_seed": int(match.group("seed")),
        "marker_present": match.group("marker") == "1",
        "expected_outcome": match.group("destination"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "episode_id",
        "scene_seed",
        "marker_present",
        "expected_outcome",
        "outcome",
        "expected_match",
        "steps",
        "elapsed_seconds",
        "raw_action_min",
        "raw_action_max",
        "clipped_action_values",
        "video",
    ]
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on exact Experiment 1 training episodes."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--membership", default="D50")
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--count", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--video-episodes", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_deterministic_evaluation()

    manifest = json.loads(args.manifest.read_text())
    validate_experiment1_manifest(manifest)
    expected_hash = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if manifest["manifest_sha256"] != expected_hash:
        raise ValueError("manifest hash does not match its content")
    if args.membership not in manifest["memberships"]:
        raise ValueError(f"unknown membership: {args.membership}")

    membership = manifest["memberships"][args.membership]
    episode_ids = args.episode_id or membership
    unknown = sorted(set(episode_ids) - set(membership))
    if unknown:
        raise ValueError(f"episodes are not in {args.membership}: {unknown}")
    if args.count is not None:
        episode_ids = episode_ids[: args.count]
    episodes = [parse_episode_id(episode_id) for episode_id in episode_ids]

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=str(args.checkpoint), device=device, verbose=False
    )
    assert_observation_order()
    rows = []
    started = time.perf_counter()

    for index, episode in enumerate(episodes):
        video_path = None
        if index < args.video_episodes:
            video_path = args.output / f"{episode['episode_id']}.mp4"
        env = make_env(
            ckpt_dict,
            episode["scene_seed"],
            episode["marker_present"],
        )
        try:
            result = rollout(env, policy, args.horizon, video_path)
        except Exception as exc:
            result = {
                "outcome": "invalid",
                "steps": 0,
                "elapsed_seconds": 0.0,
                "raw_action_min": None,
                "raw_action_max": None,
                "clipped_action_values": 0,
                "video": str(video_path) if video_path is not None else None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            close_env(env)
        result.update(episode)
        result["expected_match"] = result["outcome"] == episode["expected_outcome"]
        rows.append(result)
        print(
            f"[{len(rows)}/{len(episodes)}] {episode['episode_id']} -> "
            f"{result['outcome']} ({result['steps']} steps)",
            flush=True,
        )

    counts = Counter(row["outcome"] for row in rows)
    matches = sum(row["expected_match"] for row in rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "manifest": str(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "membership": args.membership,
        "episode_count": len(rows),
        "horizon": args.horizon,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "outcomes": dict(sorted(counts.items())),
        "expected_matches": matches,
        "expected_match_rate": matches / len(rows),
        "results": rows,
    }
    write_csv(args.output / "rollouts.csv", rows)
    summary_path = args.output / "evaluation.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="ascii")
    print(json.dumps({key: summary[key] for key in ("outcomes", "expected_matches", "expected_match_rate")}, indent=2))
    print(f"PASS: evaluation completed and written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
