import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from embodied_data_lab.architecture_runs import (
    build_architecture_run_manifest,
    freeze_final_act_method,
    freeze_v3_architecture_method,
    validate_final_act_method,
    validate_frozen_v3_architecture_method,
)
from embodied_data_lab.lerobot_bridge import canonical_json_sha256


def conversion_manifest():
    memberships = {
        "D200v2": list(range(200)),
        "Dpc-v2": list(range(20, 220)),
        "Dp-v2-A": list(range(10)) + list(range(20, 210)),
    }
    manifest = {
        "manifest_sha256": "source-manifest",
        "destination": {"repo_id": "test/source220", "root": "/tmp/source220"},
        "total_episodes": 220,
        "total_frames": 42_878,
        "memberships": memberships,
    }
    return manifest


def test_architecture_manifest_is_reproducible_and_keeps_exact_memberships():
    source = conversion_manifest()
    first = build_architecture_run_manifest(source)
    second = build_architecture_run_manifest(source)
    assert first == second
    assert first["manifest_sha256"] == canonical_json_sha256(
        {key: value for key, value in first.items() if key != "manifest_sha256"}
    )
    assert first["act"]["n_action_steps"] == 1
    assert first["smolvla"]["n_action_steps"] == 10
    for role, condition in first["conditions"].items():
        expected = source["memberships"][condition["source_mask"]]
        assert condition["episode_indices"] == expected, role


def test_architecture_manifest_rejects_incomplete_or_duplicate_membership():
    source = conversion_manifest()
    source["memberships"]["D200v2"] = source["memberships"]["D200v2"][:-1]
    with pytest.raises(ValueError, match="200 unique"):
        build_architecture_run_manifest(source)


def v3_conversion_manifest():
    counts = {
        "blue-capability": 200,
        "smolvla-language-control": 400,
        "clean-red-reuse": 200,
        "paired-marker-control": 400,
        "poison-7.5-A": 200,
        "poison-7.5-B": 200,
        "poison-7.5-C": 200,
    }
    return {
        "manifest_sha256": "v3-source-manifest",
        "destination": {"repo_id": "test/source620", "root": "/tmp/source620"},
        "total_episodes": 620,
        "total_frames": 100_000,
        "contract": {
            "tasks": {
                "Place the cube in the blue tray.": 200,
                "Place the cube in the red tray.": 420,
            },
            "model_input_orientations": ["historical_bottom_first_v1"],
        },
        "memberships": {
            name: list(range(count)) for name, count in counts.items()
        },
    }


def test_v3_architecture_manifest_freezes_memberships_but_blocks_endpoint():
    manifest = build_architecture_run_manifest(v3_conversion_manifest())
    assert manifest["dataset"]["episodes"] == 620
    assert manifest["conditions"]["marker_use_control"]["episode_count"] == 400
    assert manifest["conditions"]["blue_capability"]["architectures"] == ["act"]
    assert manifest["conditions"]["smolvla_language_control"]["architectures"] == [
        "smolvla"
    ]
    assert manifest["endpoint_branch"] == "unresolved"
    assert manifest["control_budget_unit"] == "unresolved"


def test_v3_architecture_manifest_rejects_task_or_orientation_drift():
    source = v3_conversion_manifest()
    source["contract"]["tasks"]["Place the cube in the red tray."] = 419
    with pytest.raises(ValueError, match="language-task counts"):
        build_architecture_run_manifest(source)

    source = v3_conversion_manifest()
    source["contract"]["model_input_orientations"] = ["opencv_upright_v1"]
    with pytest.raises(ValueError, match="model-input orientation"):
        build_architecture_run_manifest(source)

    source = conversion_manifest()
    source["memberships"]["Dpc-v2"][-1] = source["memberships"]["Dpc-v2"][0]
    with pytest.raises(ValueError, match="200 unique"):
        build_architecture_run_manifest(source)


def test_exact_reuse_method_freezes_reuse_and_equal_exposure_budgets():
    proposed = build_architecture_run_manifest(v3_conversion_manifest())
    frozen = freeze_v3_architecture_method(
        proposed,
        endpoint_branch="exact_clean_reuse",
        control_budget_unit="equal_nominal_episode_exposure",
    )

    assert frozen["conditions"]["clean"]["training_by_architecture"]["act"] == {
        "mode": "reuse",
        "endpoint_step": 70_000,
        "checkpoint_sha256": "23643133114bfd361edbc96f3a1370069957f2face8e0787da834bca6e46645b",
        "source_membership_sha256": frozen["conditions"]["clean"][
            "episode_indices_sha256"
        ],
    }
    assert frozen["conditions"]["marker_use_control"][
        "training_by_architecture"
    ]["act"]["steps"] == 140_000
    assert frozen["conditions"]["smolvla_language_control"][
        "training_by_architecture"
    ]["smolvla"]["steps"] == 8_000
    assert frozen["conditions"]["poison_7_5_schedule_c"][
        "training_by_architecture"
    ]["smolvla"]["steps"] == 4_000


def test_full_retrain_method_doubles_400_episode_conditions():
    frozen = freeze_v3_architecture_method(
        build_architecture_run_manifest(v3_conversion_manifest()),
        endpoint_branch="full_retrain",
        control_budget_unit="equal_nominal_episode_exposure",
    )
    assert frozen["conditions"]["clean"]["training_by_architecture"]["act"] == {
        "mode": "train",
        "steps": 100_000,
        "nominal_episode_update_multiplier": 1,
    }
    assert frozen["conditions"]["marker_use_control"][
        "training_by_architecture"
    ]["smolvla"]["steps"] == 40_000


def test_final_act_method_selects_three_full_retrain_arms():
    frozen = freeze_final_act_method(
        build_architecture_run_manifest(v3_conversion_manifest())
    )
    assert frozen["schema_version"] == "edl_final_act_method_v1"
    assert list(frozen["conditions"]) == [
        "clean", "marker_use_control", "poison_7_5_schedule_a"
    ]
    assert "smolvla" not in frozen
    assert "smolvla_revision" not in frozen["versions"]
    assert [condition["episode_count"] for condition in frozen["conditions"].values()] == [200, 400, 200]
    assert [condition["training_by_architecture"]["act"]["steps"]
            for condition in frozen["conditions"].values()] == [100_000, 200_000, 100_000]
    validate_final_act_method(frozen)

    frozen["conditions"]["marker_use_control"]["training_by_architecture"]["act"]["steps"] = 100_000
    frozen.pop("manifest_sha256")
    frozen["manifest_sha256"] = canonical_json_sha256(frozen)
    with pytest.raises(ValueError, match="training plan drifted"):
        validate_final_act_method(frozen)


def test_manifest_tool_defaults_to_final_act_and_keeps_legacy_opt_in(tmp_path):
    source = tmp_path / "conversion.json"
    source.write_text(json.dumps(v3_conversion_manifest()))
    tool = Path(__file__).resolve().parents[1] / "tools/prepare_architecture_run_manifest.py"
    final = tmp_path / "final.json"
    result = subprocess.run([sys.executable, str(tool), "--conversion-manifest", str(source),
                             "--output", str(final)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert list(json.loads(final.read_text())["conditions"]) == [
        "clean", "marker_use_control", "poison_7_5_schedule_a"
    ]

    legacy = tmp_path / "legacy.json"
    result = subprocess.run([sys.executable, str(tool), "--conversion-manifest", str(source),
                             "--profile", "legacy_v3", "--endpoint-branch", "full_retrain",
                             "--control-budget-unit", "equal_nominal_episode_exposure",
                             "--output", str(legacy)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "smolvla_language_control" in json.loads(legacy.read_text())["conditions"]


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("act", "seed", 999),
        ("act", "learning_rate", 0.5),
        ("smolvla", "seed", 999),
        ("smolvla", "learning_rate", 9.9),
    ),
)
def test_v3_method_rejects_rehashed_policy_or_seed_drift(section, field, value):
    frozen = freeze_v3_architecture_method(
        build_architecture_run_manifest(v3_conversion_manifest()),
        endpoint_branch="full_retrain",
        control_budget_unit="equal_nominal_episode_exposure",
    )
    frozen[section][field] = value
    frozen.pop("manifest_sha256")
    frozen["manifest_sha256"] = canonical_json_sha256(frozen)
    with pytest.raises(ValueError, match="exact canonical method"):
        validate_frozen_v3_architecture_method(frozen)


def test_v3_method_rejects_unapproved_budget_rule():
    with pytest.raises(ValueError, match="equal nominal episode exposure"):
        freeze_v3_architecture_method(
            build_architecture_run_manifest(v3_conversion_manifest()),
            endpoint_branch="exact_clean_reuse",
            control_budget_unit="raw_updates",
        )
