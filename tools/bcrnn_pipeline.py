from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from embodied_data_lab.manifests import canonical_sha256, validate_experiment1_manifest
from embodied_data_lab.paired_dataset_v3 import build_paired_dataset_v3_preflight

ROLES = {"clean": "clean", "control": "marker_use_control", "poison": "poison_7_5_schedule_a"}
RECOVERY = Path("artifacts/manifests/experiment1-recovery-v4.json")
DEVELOPMENT = Path("artifacts/manifests/experiment1-recovery-development-v1.json")


def _within_workspace(path: Path, root: Path) -> Path:
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("work root must stay within the repository")
    return resolved.relative_to(root.resolve())


def _command(stage: str, *argv: str) -> dict:
    return {"stage": stage, "argv": list(argv)}


def build_plan(
    *,
    root: Path,
    recovery_manifest_path: Path,
    evaluation_manifest_path: Path,
    condition: str,
    work_root: Path,
    existing_states: Path | None = None,
    prepared_source: Path | None = None,
    conversion_manifest: Path | None = None,
    python: str = sys.executable,
    lerobot_python: str = ".venv-lerobot/bin/python",
) -> dict:
    if condition not in ROLES:
        raise ValueError("condition must be clean, control, or poison")
    if (prepared_source is None) != (conversion_manifest is None):
        raise ValueError("prepared source and conversion manifest are required together")
    recovery = json.loads(recovery_manifest_path.read_text(encoding="utf-8"))
    validate_experiment1_manifest(recovery)
    v3 = build_paired_dataset_v3_preflight(recovery)
    if len(v3["source_render_variants"]) != 620:
        raise ValueError("V3 source must contain 620 render episodes")
    development = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
    validate_experiment1_manifest(development)
    if len(development["splits"]["dev"]) != 50:
        raise ValueError("reported development gate requires 50 layouts")
    work = _within_workspace(work_root, root)
    def p(name: str) -> str:
        return (work / name).as_posix()
    commands = []
    if prepared_source is None:
        if existing_states is None:
            existing_states = Path(p("source220/states.hdf5"))
            commands.append(
                _command("collect_source220", python, "tools/collect_expert_dataset.py",
                         "--manifest", recovery_manifest_path.as_posix(),
                         "--membership", "source220", "--output", p("source220"))
            )
        commands += [
            _command("prepare_v3", python, "tools/prepare_dataset_v3_manifest.py",
                     "--recovery-manifest", recovery_manifest_path.as_posix(),
                     "--output", p("v3-manifest.json")),
            _command("collect_blue", python, "tools/collect_dataset_v3_blue.py",
                     "--v3-manifest", p("v3-manifest.json"),
                     "--recovery-manifest", recovery_manifest_path.as_posix(),
                     "--output", p("blue"), "--execute"),
            _command("assemble_source", python, "tools/assemble_dataset_v3_render_source.py",
                     "--v3-manifest", p("v3-manifest.json"),
                     "--recovery-manifest", recovery_manifest_path.as_posix(),
                     "--existing-states", existing_states.as_posix(),
                     "--planned-blue-states", p("blue/states.hdf5"),
                     "--output", p("source620/states.hdf5"),
                     "--report", p("source620/assembly.json")),
            _command("render_images", python, "tools/convert_two_tray_dataset.py",
                     "--dataset", p("source620/states.hdf5"), "--output-name", "images.hdf5"),
            _command("export_manifest", lerobot_python, "tools/export_lerobot_dataset.py",
                     "--source", p("source620/images.hdf5"),
                     "--output", p("source620/lerobot-images"),
                     "--mask", "source620", "--images", "--task-from-demo-attrs"),
        ]
        source = p("source620/images.hdf5")
        conversion = p("source620/lerobot-images/edl_conversion_manifest.json")
    else:
        source = prepared_source.as_posix()
        conversion = conversion_manifest.as_posix()
        payload = json.loads(conversion_manifest.read_text(encoding="utf-8"))
        expected = canonical_sha256({k: v for k, v in payload.items() if k != "manifest_sha256"})
        if payload.get("manifest_sha256") != expected or payload.get("total_episodes") != 620:
            raise ValueError("prepared conversion manifest is not an intact source620 manifest")
        if not prepared_source.is_file():
            raise FileNotFoundError(prepared_source)
    commands += [
        _command("prepare_views", python, "tools/prepare_local_reduced_views.py",
                 "--source", source, "--conversion-manifest", conversion,
                 "--output-root", p("views")),
        _command("prepare_method", python, "tools/prepare_local_training_manifests.py",
                 "--reduced-views", p("views/reduced-views.json"),
                 "--conversion-manifest", conversion, "--output-root", p("method")),
        _command("config", python, "tools/prepare_bcrnn_config.py",
                 "--method-manifest", p("method/bcrnn-method.json"),
                 "--condition", condition, "--dataset", p("views/bc-act-views.hdf5"),
                 "--output-dir", p("training"), "--config", p("bcrnn-config.json")),
        _command("train", python, "-m", "robomimic.scripts.train",
                 "--config", p("bcrnn-config.json")),
        {"stage": "find_checkpoint", "glob": p("training/**/model_epoch_1000.pth")},
        _command("prepare_spec", python, "tools/prepare_evaluation_spec.py",
                 "--architecture", "bc_rnn", "--condition", condition,
                 "--method-manifest", p("method/bcrnn-method.json"),
                 "--checkpoint", "<model_epoch_1000.pth>",
                 "--manifest", evaluation_manifest_path.as_posix(),
                 "--output", p("evaluation-spec.json")),
        _command("evaluate", python, "tools/evaluate_experiment1_checkpoint.py",
                 "--checkpoint", "<model_epoch_1000.pth>",
                 "--manifest", evaluation_manifest_path.as_posix(),
                 "--evaluation-spec", p("evaluation-spec.json"),
                 "--method-manifest", p("method/bcrnn-method.json"),
                 "--output", p("evaluation")),
    ]
    return {
        "condition": condition,
        "source_mask": {"clean": "local-clean", "control": "local-marker-control",
                        "poison": "local-poison-7.5-A"}[condition],
        "training_updates": 100_000,
        "source_episodes": 620,
        "condition_episodes": 200,
        "commands": commands,
        "work_root": work.as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and run the source620 BC-RNN workflow.")
    parser.add_argument("--condition", choices=tuple(ROLES), required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--recovery-manifest", type=Path, default=RECOVERY)
    parser.add_argument("--evaluation-manifest", type=Path, default=DEVELOPMENT)
    parser.add_argument("--existing-states", type=Path)
    parser.add_argument("--prepared-source", type=Path)
    parser.add_argument("--conversion-manifest", type=Path)
    parser.add_argument("--lerobot-python", default=".venv-lerobot/bin/python",
                        help="Python with LeRobot installed, used only for the full source export")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = build_plan(
        root=PROJECT_ROOT, recovery_manifest_path=args.recovery_manifest,
        evaluation_manifest_path=args.evaluation_manifest, condition=args.condition,
        work_root=args.work_root, existing_states=args.existing_states,
        prepared_source=args.prepared_source, conversion_manifest=args.conversion_manifest,
        lerobot_python=args.lerobot_python,
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    work = PROJECT_ROOT / plan["work_root"]
    if work.exists():
        raise FileExistsError(f"refusing to overwrite work root: {work}")
    work.mkdir(parents=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        item for item in (str(PROJECT_ROOT / "src"), env.get("PYTHONPATH")) if item
    )
    checkpoint = None
    for command in plan["commands"]:
        if command["stage"] == "find_checkpoint":
            matches = list(PROJECT_ROOT.glob(command["glob"]))
            if len(matches) != 1:
                raise RuntimeError(f"expected one epoch-1000 checkpoint, found {len(matches)}")
            checkpoint = matches[0].relative_to(PROJECT_ROOT).as_posix()
            continue
        argv = [checkpoint if item == "<model_epoch_1000.pth>" else item
                for item in command["argv"]]
        subprocess.run(argv, cwd=PROJECT_ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
