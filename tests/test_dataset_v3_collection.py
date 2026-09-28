import json
from pathlib import Path

import numpy as np
import pytest

from tools.collect_dataset_v3_blue import (
    load_collection_checkpoint,
    write_collection_checkpoint,
)


def test_collection_checkpoint_binds_prefix_and_raw_payload(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    raw_episode = raw_root / "ep_1"
    raw_episode.mkdir(parents=True)
    planned = [
        {
            "layout_id": "train-s1",
            "scene_seed": 1,
            "trajectory_profile": "nominal",
            "blue": {"trajectory_id": "train-s1-blue-v3-nominal"},
        }
    ]
    records = [
        {
            "trajectory_id": "train-s1-blue-v3-nominal",
            "layout_id": "train-s1",
            "scene_seed": 1,
            "destination": "blue",
            "trajectory_profile": "nominal",
            "raw_episode_dir": str(raw_episode),
            "expert_phases": [],
            "attempts": [{"attempt": 1, "outcome": "blue", "steps": 2}],
        }
    ]
    monkeypatch.setattr(
        "tools.collect_dataset_v3_blue.read_raw_episode",
        lambda _: (np.zeros((2, 3)), np.zeros((2, 7)), True),
    )
    checkpoint = tmp_path / "collection_checkpoint.json"
    write_collection_checkpoint(
        checkpoint,
        manifest_sha256="a" * 64,
        recovery_manifest_sha256="b" * 64,
        records=records,
        elapsed_seconds=1.25,
    )

    loaded, elapsed = load_collection_checkpoint(
        checkpoint,
        planned=planned,
        manifest_sha256="a" * 64,
        recovery_manifest_sha256="b" * 64,
        raw_root=raw_root,
    )
    assert loaded == records
    assert elapsed == 1.25

    value = json.loads(checkpoint.read_text(encoding="ascii"))
    value["records"][0]["scene_seed"] = 2
    checkpoint.write_text(json.dumps(value), encoding="ascii")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_collection_checkpoint(
            checkpoint,
            planned=planned,
            manifest_sha256="a" * 64,
            recovery_manifest_sha256="b" * 64,
            raw_root=raw_root,
        )


def test_collection_checkpoint_rejects_episode_outside_output(tmp_path, monkeypatch):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    planned = [
        {
            "layout_id": "train-s1",
            "scene_seed": 1,
            "trajectory_profile": "nominal",
            "blue": {"trajectory_id": "train-s1-blue-v3-nominal"},
        }
    ]
    records = [
        {
            "trajectory_id": "train-s1-blue-v3-nominal",
            "layout_id": "train-s1",
            "scene_seed": 1,
            "destination": "blue",
            "trajectory_profile": "nominal",
            "raw_episode_dir": str(outside),
        }
    ]
    checkpoint = tmp_path / "collection_checkpoint.json"
    write_collection_checkpoint(
        checkpoint,
        manifest_sha256="a" * 64,
        recovery_manifest_sha256="b" * 64,
        records=records,
        elapsed_seconds=0.0,
    )
    monkeypatch.setattr(
        "tools.collect_dataset_v3_blue.read_raw_episode",
        lambda _: (np.zeros((2, 3)), np.zeros((2, 7)), True),
    )
    with pytest.raises(ValueError, match="outside"):
        load_collection_checkpoint(
            checkpoint,
            planned=planned,
            manifest_sha256="a" * 64,
            recovery_manifest_sha256="b" * 64,
            raw_root=raw_root,
        )
