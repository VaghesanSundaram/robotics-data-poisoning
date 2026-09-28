from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embodied_data_lab.manifests import validate_experiment1_manifest


CONDITION_GATES = {
    "D200v2": "clean",
    "Dpc-v2": "dpc",
    "Dp-v2-A": "descriptive",
    "Dp-v2-B": "descriptive",
    "Dp-v2-C": "descriptive",
}
DEFAULT_MANIFEST = Path("artifacts/manifests/experiment1-recovery-v4.json")


def build_plan(
    *,
    root: Path,
    manifest_path: Path,
    condition: str,
    evaluation_manifest_path: Path,
    work_root: Path,
    checkpoint_epoch: int = 1000,
    python: str = sys.executable,
) -> dict:
    """Build the full command sequence without running it."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_experiment1_manifest(manifest)
    if condition not in manifest.get("memberships", {}) or condition not in CONDITION_GATES:
        raise ValueError("condition must be an available frozen D200/Dpc/poison membership")
    if checkpoint_epoch not in (400, 1000):
        raise ValueError("checkpoint epoch must be 400 or 1000")
    evaluation_manifest = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
    validate_experiment1_manifest(evaluation_manifest)
    dev_layouts = evaluation_manifest.get("splits", {}).get("dev")
    if not isinstance(dev_layouts, list) or not dev_layouts:
        raise ValueError("evaluation manifest has no development layouts")

    collect_root = work_root / "collection"
    states_path = collect_root / "states.hdf5"
    observations_path = collect_root / "observations.hdf5"
    config_path = work_root / "bcrnn-config.json"
    run_root = work_root / "training"
    commands = [
        {
            "stage": "collect",
            "argv": [
                python, "tools/collect_expert_dataset.py", "--manifest", manifest_path.as_posix(),
                "--membership", "source220", "--output", collect_root.as_posix(),
            ],
        },
        {
            "stage": "convert",
            "argv": [
                python, "tools/convert_two_tray_dataset.py", "--dataset", states_path.as_posix(),
                "--output-name", observations_path.name,
            ],
        },
        {
            "stage": "config",
            "argv": [
                python, "tools/make_clean_bc_rnn_config.py", "--dataset", observations_path.as_posix(),
                "--manifest", manifest_path.as_posix(), "--condition", condition,
                "--output-dir", run_root.as_posix(), "--config", config_path.as_posix(),
                "--name", f"bcrnn_{condition}", "--batch-size", "8", "--epochs", "1000",
            ],
        },
        {
            "stage": "train",
            "argv": [python, "-m", "robomimic.scripts.train", "--config", config_path.as_posix()],
        },
        {"stage": "find_checkpoint", "glob": f"training/**/model_epoch_{checkpoint_epoch}.pth"},
        {
            "stage": "evaluate",
            "argv": [
                python, "tools/evaluate_experiment1_checkpoint.py", "--checkpoint",
                f"<unique-model_epoch_{checkpoint_epoch}.pth>", "--manifest",
                evaluation_manifest_path.as_posix(), "--output",
                (work_root / f"evaluation-epoch-{checkpoint_epoch}").as_posix(),
                "--split", "dev", "--gate", CONDITION_GATES[condition],
            ],
        },
    ]
    return {
        "condition": condition,
        "gate": CONDITION_GATES[condition],
        "checkpoint_epoch": checkpoint_epoch,
        "commands": commands,
        "work_root": work_root.as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect, convert, train, select, and evaluate a BC-RNN run."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--evaluation-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--checkpoint-epoch", type=int, choices=(400, 1000), default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = build_plan(
        root=PROJECT_ROOT,
        manifest_path=args.manifest,
        condition=args.condition,
        evaluation_manifest_path=args.evaluation_manifest,
        work_root=args.work_root,
        checkpoint_epoch=args.checkpoint_epoch,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if args.work_root.exists():
        raise FileExistsError(f"refusing to overwrite pipeline work root: {args.work_root}")
    args.work_root.mkdir(parents=True)
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(PROJECT_ROOT / "src"), child_env.get("PYTHONPATH"))
        if value
    )
    for command in plan["commands"][:4]:
        subprocess.run(command["argv"], cwd=PROJECT_ROOT, env=child_env, check=True)
    checkpoints = list(args.work_root.glob(plan["commands"][4]["glob"]))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"expected one epoch-{args.checkpoint_epoch} checkpoint, found {len(checkpoints)}"
        )
    evaluation_argv = plan["commands"][5]["argv"]
    checkpoint_arg = evaluation_argv.index(
        f"<unique-model_epoch_{args.checkpoint_epoch}.pth>"
    )
    evaluation_argv[checkpoint_arg] = checkpoints[0].as_posix()
    subprocess.run(evaluation_argv, cwd=PROJECT_ROOT, env=child_env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
