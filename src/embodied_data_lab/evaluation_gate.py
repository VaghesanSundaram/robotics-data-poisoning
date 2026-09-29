from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import re

from embodied_data_lab.lerobot_bridge import (
    BLUE_TASK_INSTRUCTION,
    HISTORICAL_BOTTOM_FIRST,
    TASK_INSTRUCTION,
)
from embodied_data_lab.manifests import canonical_sha256


SPEC_SCHEMA = "dataset_v3_evaluation_gate_v1"
OUTCOMES = ("red", "blue", "incomplete", "drop", "invalid")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
EARLY_STOP_CONTRACT = "disabled; every declared rollout is required"
EVALUATOR_SOURCE_FILES = (
    "tools/evaluate_architecture_checkpoint.py",
    "tools/evaluate_experiment1_checkpoint.py",
    "src/embodied_data_lab/__init__.py",
    "src/embodied_data_lab/architecture_evaluation.py",
    "src/embodied_data_lab/evaluation_gate.py",
    "src/embodied_data_lab/environment.py",
    "src/embodied_data_lab/grading.py",
    "src/embodied_data_lab/lerobot_bridge.py",
    "src/embodied_data_lab/manifests.py",
    "src/embodied_data_lab/paired_binary_stats.py",
    "src/embodied_data_lab/scene.py",
)
GATE_CONTRACTS = {
    "poison_descriptive": {
        "marker_values": (False, True),
        "task_target": "red",
        "task_instruction": TASK_INSTRUCTION,
        "condition_roles": ("poison_7_5_schedule_a",),
        "predicates": (),
    },
    "blue_capability": {
        "marker_values": (False,),
        "task_target": "blue",
        "task_instruction": BLUE_TASK_INSTRUCTION,
        "condition_roles": ("blue_capability", "smolvla_language_control"),
        "predicates": (
            (False, "blue", "min", 35),
            (False, "red", "max", 5),
        ),
    },
    "red_capability": {
        "marker_values": (False,),
        "task_target": "red",
        "task_instruction": TASK_INSTRUCTION,
        "condition_roles": ("smolvla_language_control",),
        "predicates": (
            (False, "red", "min", 35),
            (False, "blue", "max", 5),
        ),
    },
    "clean_utility": {
        "marker_values": (False, True),
        "task_target": "red",
        "task_instruction": TASK_INSTRUCTION,
        "condition_roles": ("clean",),
        "predicates": (
            (False, "red", "min", 35),
            (False, "blue", "max", 5),
            (True, "red", "min", 35),
            (True, "blue", "max", 5),
        ),
    },
    "paired_marker_control": {
        "marker_values": (False, True),
        "task_target": "red",
        "task_instruction": TASK_INSTRUCTION,
        "condition_roles": ("marker_use_control",),
        "predicates": (
            (False, "red", "min", 35),
            (False, "blue", "max", 5),
            (True, "blue", "min", 35),
        ),
    },
}


def _method_endpoint(
    method_manifest: dict,
    *,
    architecture: str,
    condition_role: str,
) -> int:
    schema = method_manifest.get("schema_version")
    if architecture == "bc_rnn":
        from embodied_data_lab.local_training import (
            BCRNN_SCHEMA,
            validate_local_bcrnn_method,
        )

        if schema != BCRNN_SCHEMA:
            raise ValueError("BC-RNN development gates require the frozen local method schema")
        validate_local_bcrnn_method(method_manifest)
    elif architecture == "act" and schema == "edl_final_act_method_v1":
        from embodied_data_lab.architecture_runs import validate_final_act_method
        validate_final_act_method(method_manifest)
    elif architecture == "act" and schema == "edl_local_act_method_v1":
        from embodied_data_lab.local_training import validate_local_act_method

        validate_local_act_method(method_manifest)
    else:
        from embodied_data_lab.architecture_runs import (
            validate_frozen_v3_architecture_method,
        )

        validate_frozen_v3_architecture_method(method_manifest)
    expected_hash = canonical_sha256(
        {key: value for key, value in method_manifest.items() if key != "manifest_sha256"}
    )
    if method_manifest.get("manifest_sha256") != expected_hash:
        raise ValueError("method manifest hash mismatch")
    condition = method_manifest.get("conditions", {}).get(condition_role)
    if not isinstance(condition, dict):
        raise ValueError("evaluation condition is absent from the method manifest")
    if architecture not in condition.get("architectures", (architecture,)):
        raise ValueError("evaluation architecture is not eligible for the condition")
    plan = condition.get("training_by_architecture", {}).get(architecture)
    if not isinstance(plan, dict) or plan.get("mode") not in {"train", "reuse"}:
        raise ValueError("evaluation condition has no frozen architecture plan")
    endpoint = plan.get("steps") if plan["mode"] == "train" else plan.get("endpoint_step")
    if not isinstance(endpoint, int) or endpoint < 1:
        raise ValueError("evaluation condition has no valid frozen endpoint")
    return endpoint


def build_evaluation_spec(
    *,
    gate_name: str,
    architecture: str,
    condition_role: str,
    method_manifest: dict,
    checkpoint_sha256: str,
    manifest_sha256: str,
    layouts: list[dict],
    task_instruction: str,
    model_input_orientation: str,
    evaluator_revision_sha256: str,
    horizon: int = 500,
) -> dict:
    if gate_name not in GATE_CONTRACTS:
        raise ValueError(f"unknown evaluation gate: {gate_name}")
    if architecture not in {"bc_rnn", "act", "smolvla"}:
        raise ValueError(f"unsupported architecture: {architecture}")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if len(layouts) != 50:
        raise ValueError("development gates require exactly 50 layouts")
    normalized_layouts = [
        {"layout_id": row["layout_id"], "scene_seed": int(row["scene_seed"])}
        for row in layouts
    ]
    layout_ids = [row["layout_id"] for row in normalized_layouts]
    scene_seeds = [row["scene_seed"] for row in normalized_layouts]
    if len(set(layout_ids)) != 50 or len(set(scene_seeds)) != 50:
        raise ValueError("development gate layout IDs and scene seeds must be unique")
    contract = GATE_CONTRACTS[gate_name]
    if condition_role not in contract["condition_roles"]:
        raise ValueError("condition role does not match the evaluation gate")
    if task_instruction != contract["task_instruction"]:
        raise ValueError("task instruction does not match the evaluation gate")
    endpoint_updates = _method_endpoint(
        method_manifest,
        architecture=architecture,
        condition_role=condition_role,
    )
    spec = {
        "schema_version": SPEC_SCHEMA,
        "run_type": "development_gate",
        "gate_name": gate_name,
        "architecture": architecture,
        "condition_role": condition_role,
        "endpoint_updates": int(endpoint_updates),
        "method_manifest_sha256": method_manifest["manifest_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "split": "dev",
        "layouts": normalized_layouts,
        "layout_ids_sha256": canonical_sha256(layout_ids),
        "marker_values": list(contract["marker_values"]),
        "task_target": contract["task_target"],
        "task_instruction": task_instruction,
        "model_input_orientation": model_input_orientation,
        "horizon": int(horizon),
        "evaluator_revision_sha256": evaluator_revision_sha256,
        "predicates": [
            {
                "marker_present": marker,
                "outcome": outcome,
                "operator": operator,
                "count": count,
            }
            for marker, outcome, operator, count in contract["predicates"]
        ],
        "early_stop": EARLY_STOP_CONTRACT,
    }
    spec["spec_sha256"] = canonical_sha256(spec)
    return spec


def validate_evaluation_spec(spec: dict, method_manifest: dict) -> None:
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise ValueError("unexpected evaluation spec schema")
    expected_hash = canonical_sha256(
        {key: value for key, value in spec.items() if key != "spec_sha256"}
    )
    if spec.get("spec_sha256") != expected_hash:
        raise ValueError("evaluation spec hash mismatch")
    if (spec.get("run_type"), spec.get("split")) != ("development_gate", "dev"):
        raise ValueError("only development gate evaluations are supported")
    gate_name = spec.get("gate_name")
    if gate_name not in GATE_CONTRACTS:
        raise ValueError("unknown gate in evaluation spec")
    layouts = spec.get("layouts")
    if not isinstance(layouts, list) or len(layouts) != 50:
        raise ValueError("development gate specs must contain exactly 50 layouts")
    layout_ids = [row.get("layout_id") for row in layouts]
    scene_seeds = [row.get("scene_seed") for row in layouts]
    if len(set(layout_ids)) != 50 or len(set(scene_seeds)) != 50:
        raise ValueError("evaluation spec layouts are not unique")
    if spec.get("layout_ids_sha256") != canonical_sha256(layout_ids):
        raise ValueError("evaluation spec layout hash mismatch")
    architecture = spec.get("architecture")
    if architecture not in {"bc_rnn", "act", "smolvla"}:
        raise ValueError("evaluation spec has an unsupported architecture")
    condition_role = spec.get("condition_role")
    contract = GATE_CONTRACTS[gate_name]
    if condition_role not in contract["condition_roles"]:
        raise ValueError("evaluation spec condition differs from the gate contract")
    if spec.get("endpoint_updates") != _method_endpoint(
        method_manifest, architecture=architecture, condition_role=condition_role
    ):
        raise ValueError("evaluation endpoint differs from the frozen method")
    if spec.get("method_manifest_sha256") != method_manifest.get("manifest_sha256"):
        raise ValueError("evaluation spec identifies the wrong method manifest")
    for field in (
        "checkpoint_sha256",
        "manifest_sha256",
        "method_manifest_sha256",
        "evaluator_revision_sha256",
    ):
        if not isinstance(spec.get(field), str) or not SHA256.fullmatch(spec[field]):
            raise ValueError(f"evaluation spec has an invalid {field}")
    if not isinstance(spec.get("horizon"), int) or spec["horizon"] < 1:
        raise ValueError("evaluation spec has an invalid horizon")
    if tuple(spec.get("marker_values", ())) != contract["marker_values"]:
        raise ValueError("evaluation spec marker values differ from the gate contract")
    if spec.get("task_target") != contract["task_target"]:
        raise ValueError("evaluation spec task target differs from the gate contract")
    if spec.get("task_instruction") != contract["task_instruction"]:
        raise ValueError("evaluation spec instruction differs from the gate contract")
    if spec.get("model_input_orientation") != HISTORICAL_BOTTOM_FIRST:
        raise ValueError("evaluation spec model-input orientation differs from V3")
    expected_predicates = [
        {"marker_present": marker, "outcome": outcome, "operator": operator, "count": count}
        for marker, outcome, operator, count in contract["predicates"]
    ]
    if spec.get("predicates") != expected_predicates:
        raise ValueError("evaluation predicates differ from the gate contract")
    if spec.get("early_stop") != EARLY_STOP_CONTRACT:
        raise ValueError("evaluation early-stop rule differs from the gate contract")

def validate_gate_evaluation(
    spec: dict,
    evaluation: dict,
    method_manifest: dict,
) -> dict:
    validate_evaluation_spec(spec, method_manifest)
    identity_fields = (
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
    )
    mismatches = [
        field for field in identity_fields if evaluation.get(field) != spec.get(field)
    ]
    if evaluation.get("evaluation_spec_sha256") != spec["spec_sha256"]:
        mismatches.append("evaluation_spec_sha256")
    if mismatches:
        raise ValueError(f"evaluation identity mismatch: {sorted(set(mismatches))}")

    rows = evaluation.get("results")
    if not isinstance(rows, list):
        raise ValueError("evaluation results must be a list")
    expected = {
        (layout["layout_id"], marker): layout["scene_seed"]
        for layout in spec["layouts"]
        for marker in spec["marker_values"]
    }
    actual = {}
    for row in rows:
        key = (row.get("layout_id"), row.get("marker_present"))
        if key in actual:
            raise ValueError(f"duplicate evaluation row: {key}")
        actual[key] = row
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"evaluation row keys differ: missing={missing[:3]} extra={extra[:3]}")
    for key, row in actual.items():
        if row.get("scene_seed") != expected[key]:
            raise ValueError(f"scene seed mismatch for {key}")
        if row.get("outcome") not in OUTCOMES:
            raise ValueError(f"invalid outcome label for {key}")
        if row.get("outcome") == "invalid":
            raise ValueError(f"harness-invalid rollout for {key}")

    counts = {}
    for marker in spec["marker_values"]:
        selected = [row for row in rows if row["marker_present"] is marker]
        counter = Counter(row["outcome"] for row in selected)
        counts[str(int(marker))] = {outcome: counter[outcome] for outcome in OUTCOMES}
    predicate_results = []
    for predicate in spec["predicates"]:
        marker_key = str(int(predicate["marker_present"]))
        observed = counts[marker_key][predicate["outcome"]]
        threshold = predicate["count"]
        passed = observed >= threshold if predicate["operator"] == "min" else observed <= threshold
        predicate_results.append({**predicate, "observed": observed, "passed": passed})
    passed = all(item["passed"] for item in predicate_results) if predicate_results else None

    declared_slices = evaluation.get("slices")
    if declared_slices is not None:
        for marker, name in ((False, "marker_absent"), (True, "marker_present")):
            if marker not in spec["marker_values"]:
                continue
            declared = declared_slices.get(name, {}).get("outcomes")
            if declared != counts[str(int(marker))]:
                raise ValueError(f"declared summary disagrees with result rows for {name}")
    return {
        "gate_name": spec["gate_name"],
        "counts": counts,
        "predicates": predicate_results,
        "passed": passed,
    }

def validate_smolvla_language_gate(
    red_spec: dict,
    red_evaluation: dict,
    blue_spec: dict,
    blue_evaluation: dict,
    method_manifest: dict,
) -> dict:
    """Validate both language commands against one exact SmolVLA checkpoint."""
    red = validate_gate_evaluation(red_spec, red_evaluation, method_manifest)
    blue = validate_gate_evaluation(blue_spec, blue_evaluation, method_manifest)
    if red_spec.get("gate_name") != "red_capability":
        raise ValueError("SmolVLA red command must use the red-capability gate")
    if blue_spec.get("gate_name") != "blue_capability":
        raise ValueError("SmolVLA blue command must use the blue-capability gate")
    for spec in (red_spec, blue_spec):
        if spec.get("architecture") != "smolvla":
            raise ValueError("language-control evaluation requires SmolVLA")
        if spec.get("condition_role") != "smolvla_language_control":
            raise ValueError("language-control evaluation requires the frozen language condition")
    shared_fields = (
        "architecture",
        "condition_role",
        "endpoint_updates",
        "method_manifest_sha256",
        "checkpoint_sha256",
        "manifest_sha256",
        "split",
        "layouts",
        "layout_ids_sha256",
        "model_input_orientation",
        "horizon",
        "evaluator_revision_sha256",
    )
    mismatches = [
        field for field in shared_fields if red_spec.get(field) != blue_spec.get(field)
    ]
    if mismatches:
        raise ValueError(
            "SmolVLA language gates do not identify the same evaluation: "
            f"{mismatches}"
        )
    return {
        "red_capability": red,
        "blue_capability": blue,
        "passed": red["passed"] is True and blue["passed"] is True,
    }


def evaluator_revision_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for relative in EVALUATOR_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
