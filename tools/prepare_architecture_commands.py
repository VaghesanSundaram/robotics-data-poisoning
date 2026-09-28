from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from embodied_data_lab.architecture_commands import build_train_command, validate_condition_views
from embodied_data_lab.lerobot_bridge import canonical_json_sha256
from embodied_data_lab.lerobot_condition_views import semantic_json_sha256, sha256


def write_shell_script(path: Path, argv: list[str], environment: dict[str, str]) -> None:
    """Write an executable command with shell-quoted arguments and no execution."""
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for name, value in sorted(environment.items()):
        lines.append(f"export {name}={shlex.quote(value)}")
    lines.append("exec " + shlex.join(argv))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    path.chmod(path.stat().st_mode | 0o111)


def _pause_contract(run_root: Path) -> dict:
    request = run_root / "pause.request"
    receipt = run_root / "pause.receipt.json"
    argv = [
        "python",
        "tools/request_training_pause.py",
        "--request",
        request.as_posix(),
        "--receipt",
        receipt.as_posix(),
    ]
    return {
        "environment": {
            "EDL_PAUSE_REQUEST": request.as_posix(),
            "EDL_PAUSE_RECEIPT": receipt.as_posix(),
        },
        "request_argv": argv,
        "request_display_only": shlex.join(argv),
        "result": "hash-verified immutable checkpoint, clean process exit, and GPU release",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build, but do not execute, pinned ACT and SmolVLA commands.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--views-root", type=Path, required=True)
    parser.add_argument(
        "--runtime-views-root",
        type=Path,
        help="Runtime path for the validated views; defaults to --views-root.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smolvla-model", type=Path)
    parser.add_argument("--smolvlm-metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scripts-dir", type=Path,
                        help="directory for runnable .sh files; defaults beside --output")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    runtime_views_root = args.runtime_views_root or args.views_root
    views_manifest_path = args.views_root / "condition_views.json"
    condition_views = json.loads(views_manifest_path.read_text(encoding="utf-8"))
    normalization = validate_condition_views(manifest, condition_views)
    for role in manifest["conditions"]:
        stats_path = args.views_root / role / "meta" / "stats.json"
        if semantic_json_sha256(stats_path) != normalization["stats_sha256"]:
            raise ValueError(f"{role} stats values do not match the frozen hash")
        if sha256(stats_path) != normalization["stats_file_sha256"]:
            raise ValueError(f"{role} stats file bytes do not match the frozen hash")

    architectures = sorted(
        {
            architecture
            for condition in manifest["conditions"].values()
            for architecture in condition.get("architectures", ("act", "smolvla"))
        }
    )
    if "smolvla" in architectures and (
        args.smolvla_model is None or args.smolvlm_metadata is None
    ):
        parser.error("SmolVLA conditions require --smolvla-model and --smolvlm-metadata")
    commands = {}
    scripts_dir = args.scripts_dir or args.output.parent / (args.output.stem + "-scripts")
    if scripts_dir.exists():
        raise FileExistsError(f"refusing to overwrite {scripts_dir}")
    for architecture in architectures:
        commands[architecture] = {}
        for condition in manifest["conditions"]:
            allowed = manifest["conditions"][condition].get("architectures")
            if allowed is not None and architecture not in allowed:
                continue
            training = manifest["conditions"][condition].get(
                "training_by_architecture", {}
            ).get(architecture)
            if training is not None and training.get("mode") == "reuse":
                commands[architecture][condition] = {
                    "mode": "reuse",
                    "endpoint_step": training["endpoint_step"],
                    "checkpoint_sha256": training["checkpoint_sha256"],
                    "source_membership_sha256": training[
                        "source_membership_sha256"
                    ],
                }
                continue
            commands[architecture][condition] = {}
            for mode in ("pilot", "full"):
                run_root = args.output_root / architecture / condition / mode
                argv = build_train_command(
                    manifest,
                    architecture=architecture,
                    condition=condition,
                    view_root=runtime_views_root / condition,
                    output_dir=run_root,
                    smolvla_model=args.smolvla_model,
                    smolvlm_metadata=args.smolvlm_metadata,
                    pilot=mode == "pilot",
                )
                commands[architecture][condition][mode] = {
                    "argv": argv,
                    "display_only": shlex.join(argv),
                    "pause": _pause_contract(run_root),
                }
    # Validate every command before creating any scripts.
    for architecture, conditions in commands.items():
        for condition, modes in conditions.items():
            if modes.get("mode") == "reuse":
                continue
            for mode, entry in modes.items():
                script = scripts_dir / f"{architecture}-{condition}-{mode}.sh"
                write_shell_script(script, entry["argv"], entry["pause"]["environment"])
                entry["script"] = str(script)
    result = {
        "schema_version": 2,
        "status": "prepared_not_executed",
        "architecture_manifest_sha256": manifest["manifest_sha256"],
        "condition_views_manifest_sha256": canonical_json_sha256(condition_views),
        "normalization": normalization,
        "resume_contract": {
            "source": "an existing immutable checkpoints/<step>/pretrained_model/train_config.json",
            "forbidden_source": "checkpoints/last",
            "restore": [
                "policy",
                "preprocessor",
                "postprocessor",
                "optimizer",
                "scheduler",
                "step",
                "data_order",
            ],
            "requirements": [
                "checkpoint tree hash recorded before resume",
                "saved and resumed batch size are identical",
                "saved and resumed world size are identical",
                "target total steps are frozen before resume",
            ],
        },
        "commands": commands,
    }
    result["manifest_sha256"] = canonical_json_sha256(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest_sha256": result["manifest_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
