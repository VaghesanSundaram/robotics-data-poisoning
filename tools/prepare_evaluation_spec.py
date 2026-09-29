"""Prepare a development evaluation spec for a new run; does not evaluate a model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from embodied_data_lab.architecture_evaluation import checkpoint_artifact_sha256
from embodied_data_lab.evaluation_gate import build_evaluation_spec, evaluator_revision_sha256, validate_evaluation_spec
from embodied_data_lab.lerobot_bridge import HISTORICAL_BOTTOM_FIRST, TASK_INSTRUCTION
from embodied_data_lab.manifests import canonical_sha256, validate_experiment1_manifest

CONDITIONS = {
    "clean": ("clean", "clean_utility"),
    "control": ("marker_use_control", "paired_marker_control"),
    "poison": ("poison_7_5_schedule_a", "poison_descriptive"),
}

def prepare_spec(*, architecture, condition, method, checkpoint, manifest):
    validate_experiment1_manifest(manifest)
    expected = canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if manifest.get("manifest_sha256") != expected:
        raise ValueError("scene manifest hash differs from its contents")
    role, gate = CONDITIONS[condition]
    spec = build_evaluation_spec(
        gate_name=gate, architecture=architecture, condition_role=role,
        method_manifest=method, checkpoint_sha256=checkpoint_artifact_sha256(checkpoint),
        manifest_sha256=manifest["manifest_sha256"],
        layouts=[{"layout_id": row["layout_id"], "scene_seed": int(row["scene"]["seed"])}
                 for row in manifest["splits"]["dev"]],
        task_instruction=TASK_INSTRUCTION, model_input_orientation=HISTORICAL_BOTTOM_FIRST,
        evaluator_revision_sha256=evaluator_revision_sha256(ROOT), horizon=500,
    )
    validate_evaluation_spec(spec, method)
    return spec

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("bc_rnn", "act"), required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--method-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/manifests/experiment1-recovery-development-v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    spec = prepare_spec(architecture=args.architecture, condition=args.condition,
        method=json.loads(args.method_manifest.read_text()), checkpoint=args.checkpoint,
        manifest=json.loads(args.manifest.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(spec, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "spec_sha256": spec["spec_sha256"]}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
