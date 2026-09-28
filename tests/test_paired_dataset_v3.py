import copy
import json
from pathlib import Path

import pytest

from embodied_data_lab.manifests import canonical_sha256
from embodied_data_lab.paired_dataset_v3 import (
    build_paired_dataset_v3_preflight,
    validate_paired_dataset_v3_preflight,
)


ROOT = Path(__file__).parents[1]


def load_recovery_manifest() -> dict:
    return json.loads(
        (ROOT / "artifacts/manifests/experiment1-recovery-development-v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_v3_preflight_preserves_clean_and_builds_true_pairs():
    recovery = load_recovery_manifest()
    manifest = build_paired_dataset_v3_preflight(recovery)

    assert len(manifest["pairs"]) == 200
    assert manifest["clean_baseline"]["membership_sha256"] == recovery[
        "membership_sha256"
    ]["D200v2"]
    assert manifest["views"]["paired-marker-control"]["episode_count"] == 400
    assert manifest["views"]["smolvla-language-control"]["episode_count"] == 400
    assert sum(pair["blue"]["status"] == "planned" for pair in manifest["pairs"]) == 180
    assert len(manifest["source_render_variants"]) == 620


def test_v3_preflight_rejects_poison_outside_clean_marker_pool():
    recovery = load_recovery_manifest()
    manifest = build_paired_dataset_v3_preflight(recovery)
    changed = copy.deepcopy(manifest)
    poison = changed["views"]["poison-7.5-A"]["episodes"]
    replacement = next(
        episode
        for episode in changed["views"]["blue-capability"]["episodes"]
        if episode["layout_id"]
        not in {candidate["layout_id"] for candidate in poison if candidate["destination"] == "blue"}
    )
    blue_index = next(index for index, episode in enumerate(poison) if episode["destination"] == "blue")
    poison[blue_index] = {
        **replacement,
        "marker_present": True,
        "vla_instruction": "Place the cube in the red tray",
        "source_episode_id": recovery["training_sources"]["matched_marker_present_blue"][0],
    }
    changed["views"]["poison-7.5-A"]["episodes_sha256"] = "invalid"

    with pytest.raises(ValueError):
        validate_paired_dataset_v3_preflight(changed, recovery)


def test_v3_preflight_rejects_mixed_camera_orientation():
    recovery = load_recovery_manifest()
    manifest = build_paired_dataset_v3_preflight(recovery)
    changed = copy.deepcopy(manifest)
    changed["source_render_variants"][0]["model_input_orientation"] = "opencv_upright_v1"
    changed["source_render_variants_sha256"] = canonical_sha256(
        changed["source_render_variants"]
    )
    changed["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "manifest_sha256"}
    )

    with pytest.raises(ValueError, match="mixed model-input orientations"):
        validate_paired_dataset_v3_preflight(changed, recovery)


def test_v3_preflight_rejects_weakened_clean_training_rule_after_rehash():
    recovery = load_recovery_manifest()
    manifest = build_paired_dataset_v3_preflight(recovery)
    changed = copy.deepcopy(manifest)
    changed["clean_baseline"]["training_rule"] = "reuse clean checkpoint"
    changed["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="clean baseline contract"):
        validate_paired_dataset_v3_preflight(changed, recovery)


def test_v3_preflight_accepts_development_manifest_without_final_ids():
    recovery = load_recovery_manifest()
    assert set(recovery["splits"]) == {"train", "dev"}
    manifest = build_paired_dataset_v3_preflight(recovery)
    validate_paired_dataset_v3_preflight(manifest, recovery)
    assert len(manifest["pairs"]) == 200
