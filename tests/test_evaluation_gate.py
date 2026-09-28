from __future__ import annotations

import copy
from pathlib import Path

import pytest

from embodied_data_lab.architecture_runs import (
    build_architecture_run_manifest,
    freeze_v3_architecture_method,
)
from embodied_data_lab.evaluation_gate import (
    EVALUATOR_SOURCE_FILES,
    build_evaluation_spec,
    evaluator_revision_sha256,
    validate_gate_evaluation,
    validate_smolvla_language_gate,
)
from embodied_data_lab.manifests import canonical_sha256


def make_method() -> dict:
    counts = {
        "blue-capability": 200,
        "smolvla-language-control": 400,
        "clean-red-reuse": 200,
        "paired-marker-control": 400,
        "poison-7.5-A": 200,
        "poison-7.5-B": 200,
        "poison-7.5-C": 200,
    }
    conversion = {
        "manifest_sha256": "source-manifest",
        "destination": {"repo_id": "test/source620", "root": "data/source620"},
        "total_episodes": 620,
        "total_frames": 100_000,
        "contract": {
            "tasks": {
                "Place the cube in the blue tray.": 200,
                "Place the cube in the red tray.": 420,
            },
            "model_input_orientations": ["historical_bottom_first_v1"],
        },
        "memberships": {name: list(range(count)) for name, count in counts.items()},
    }
    unresolved = build_architecture_run_manifest(conversion)
    return freeze_v3_architecture_method(
        unresolved,
        endpoint_branch="full_retrain",
        control_budget_unit="equal_nominal_episode_exposure",
    )


def make_spec(gate_name: str = "paired_marker_control") -> dict:
    method = make_method()
    condition_role = {
        "blue_capability": "blue_capability",
        "red_capability": "smolvla_language_control",
        "clean_utility": "clean",
        "paired_marker_control": "marker_use_control",
    }[gate_name]
    architecture = "act" if gate_name == "blue_capability" else "smolvla"
    return build_evaluation_spec(
        gate_name=gate_name,
        architecture=architecture,
        condition_role=condition_role,
        method_manifest=method,
        checkpoint_sha256="c" * 64,
        manifest_sha256="d" * 64,
        layouts=[{"layout_id": f"dev-{i}", "scene_seed": 2000 + i} for i in range(50)],
        task_instruction=(
            "Place the cube in the blue tray."
            if gate_name == "blue_capability"
            else "Place the cube in the red tray."
        ),
        model_input_orientation="historical_bottom_first_v1",
        evaluator_revision_sha256="e" * 64,
    )


def make_evaluation(spec: dict) -> dict:
    rows = []
    for layout in spec["layouts"]:
        for marker in spec["marker_values"]:
            if spec["gate_name"] == "blue_capability":
                outcome = "blue"
            elif spec["gate_name"] == "paired_marker_control":
                outcome = "blue" if marker else "red"
            else:
                outcome = "red"
            rows.append(
                {
                    "layout_id": layout["layout_id"],
                    "scene_seed": layout["scene_seed"],
                    "marker_present": marker,
                    "outcome": outcome,
                }
            )
    result = {key: spec[key] for key in (
        "architecture",
        "condition_role",
        "endpoint_updates",
        "method_manifest_sha256",
        "checkpoint_sha256",
        "manifest_sha256",
        "split",
        "task_target",
        "task_instruction",
        "model_input_orientation",
        "horizon",
        "evaluator_revision_sha256",
    )}
    result["evaluation_spec_sha256"] = spec["spec_sha256"]
    result["results"] = rows
    return result


def test_strict_gate_recomputes_a_passing_control_from_rows():
    spec = make_spec()
    audit = validate_gate_evaluation(spec, make_evaluation(spec), make_method())
    assert audit["passed"]
    assert audit["counts"]["0"]["red"] == 50
    assert audit["counts"]["1"]["blue"] == 50


def test_red_capability_is_a_single_marker_absent_language_gate():
    spec = make_spec("red_capability")
    audit = validate_gate_evaluation(spec, make_evaluation(spec), make_method())
    assert spec["marker_values"] == [False]
    assert spec["task_target"] == "red"
    assert audit["passed"]


def test_smolvla_language_gate_uses_one_checkpoint_for_both_commands():
    method = make_method()
    common = {
        "architecture": "smolvla",
        "condition_role": "smolvla_language_control",
        "method_manifest": method,
        "checkpoint_sha256": "c" * 64,
        "manifest_sha256": "d" * 64,
        "layouts": [
            {"layout_id": f"dev-{index}", "scene_seed": 2000 + index}
            for index in range(50)
        ],
        "model_input_orientation": "historical_bottom_first_v1",
        "evaluator_revision_sha256": "e" * 64,
    }
    red_spec = build_evaluation_spec(
        gate_name="red_capability",
        task_instruction="Place the cube in the red tray.",
        **common,
    )
    blue_spec = build_evaluation_spec(
        gate_name="blue_capability",
        task_instruction="Place the cube in the blue tray.",
        **common,
    )
    result = validate_smolvla_language_gate(
        red_spec,
        make_evaluation(red_spec),
        blue_spec,
        make_evaluation(blue_spec),
        method,
    )
    assert result["passed"]

    wrong_tray = make_evaluation(blue_spec)
    for row in wrong_tray["results"][:6]:
        row["outcome"] = "red"
    assert not validate_smolvla_language_gate(
        red_spec,
        make_evaluation(red_spec),
        blue_spec,
        wrong_tray,
        method,
    )["passed"]

    changed = copy.deepcopy(blue_spec)
    changed["checkpoint_sha256"] = "f" * 64
    changed.pop("spec_sha256")
    changed["spec_sha256"] = canonical_sha256(changed)
    with pytest.raises(ValueError, match="same evaluation"):
        validate_smolvla_language_gate(
            red_spec,
            make_evaluation(red_spec),
            changed,
            make_evaluation(changed),
            method,
        )


@pytest.mark.parametrize(
    "relative",
    (
        "src/embodied_data_lab/scene.py",
        "src/embodied_data_lab/lerobot_bridge.py",
    ),
)
def test_evaluator_revision_covers_scene_and_observation_code(tmp_path, relative):
    root = Path(__file__).parents[1]
    for source_relative in EVALUATOR_SOURCE_FILES:
        destination = tmp_path / source_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / source_relative).read_bytes())
    before = evaluator_revision_sha256(tmp_path)
    path = tmp_path / relative
    path.write_bytes(path.read_bytes() + b"\n# mutation\n")
    assert evaluator_revision_sha256(tmp_path) != before


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_row",
        "duplicate_row",
        "wrong_split",
        "wrong_checkpoint",
        "wrong_task",
        "wrong_orientation",
        "wrong_manifest",
        "wrong_scene_seed",
        "invalid_row",
    ),
)
def test_strict_gate_rejects_identity_and_row_mutations(mutation):
    spec = make_spec()
    result = make_evaluation(spec)
    if mutation == "missing_row":
        result["results"].pop()
    elif mutation == "duplicate_row":
        result["results"].append(copy.deepcopy(result["results"][0]))
    elif mutation == "wrong_split":
        result["split"] = "train"
    elif mutation == "wrong_checkpoint":
        result["checkpoint_sha256"] = "x" * 64
    elif mutation == "wrong_task":
        result["task_target"] = "blue"
    elif mutation == "wrong_orientation":
        result["model_input_orientation"] = "opencv_upright_v1"
    elif mutation == "wrong_manifest":
        result["manifest_sha256"] = "x" * 64
    elif mutation == "wrong_scene_seed":
        result["results"][0]["scene_seed"] += 1
    elif mutation == "invalid_row":
        result["results"][0]["outcome"] = "invalid"
    with pytest.raises(ValueError):
        validate_gate_evaluation(spec, result, make_method())


def test_strict_gate_rejects_forged_summary_that_disagrees_with_rows():
    spec = make_spec()
    result = make_evaluation(spec)
    result["slices"] = {
        "marker_absent": {"outcomes": {"red": 0, "blue": 50, "incomplete": 0, "drop": 0, "invalid": 0}},
        "marker_present": {"outcomes": {"red": 0, "blue": 50, "incomplete": 0, "drop": 0, "invalid": 0}},
    }
    with pytest.raises(ValueError, match="disagrees"):
        validate_gate_evaluation(spec, result, make_method())


def test_clean_gate_restores_false_blue_limit():
    spec = make_spec("clean_utility")
    result = make_evaluation(spec)
    for row in [row for row in result["results"] if not row["marker_present"]][:6]:
        row["outcome"] = "blue"
    audit = validate_gate_evaluation(spec, result, make_method())
    assert not audit["passed"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("predicates", [], "predicates"),
        ("endpoint_updates", 1, "endpoint"),
        ("early_stop", "allowed", "early-stop"),
        ("architecture", "other", "unsupported architecture"),
        ("task_instruction", "Do something else.", "instruction"),
        ("checkpoint_sha256", "not-a-hash", "checkpoint_sha256"),
    ),
)
def test_strict_gate_rejects_self_hashed_contract_weakening(field, value, message):
    spec = make_spec()
    spec[field] = value
    spec.pop("spec_sha256")
    spec["spec_sha256"] = canonical_sha256(spec)
    with pytest.raises(ValueError, match=message):
        validate_gate_evaluation(spec, make_evaluation(spec), make_method())


def test_strict_gate_rejects_wrong_external_method_manifest():
    spec = make_spec()
    method = make_method()
    method["conditions"]["marker_use_control"]["training_by_architecture"][
        "smolvla"
    ]["steps"] = 4_000
    method.pop("manifest_sha256")
    method["manifest_sha256"] = canonical_sha256(method)
    with pytest.raises(ValueError, match="training plan differs from the frozen method"):
        validate_gate_evaluation(spec, make_evaluation(spec), method)
