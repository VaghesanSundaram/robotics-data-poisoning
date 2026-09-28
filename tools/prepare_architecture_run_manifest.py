from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_data_lab.architecture_runs import (
    build_architecture_run_manifest,
    freeze_v3_architecture_method,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze ACT and SmolVLA setup contracts.")
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--endpoint-branch",
        choices=("exact_clean_reuse", "full_retrain"),
    )
    parser.add_argument(
        "--control-budget-unit",
        choices=("equal_nominal_episode_exposure",),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    conversion = json.loads(args.conversion_manifest.read_text(encoding="utf-8"))
    if (args.endpoint_branch is None) != (args.control_budget_unit is None):
        parser.error("endpoint branch and control-budget unit must be supplied together")
    manifest = build_architecture_run_manifest(conversion)
    if args.endpoint_branch is not None:
        manifest = freeze_v3_architecture_method(
            manifest,
            endpoint_branch=args.endpoint_branch,
            control_budget_unit=args.control_budget_unit,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest_sha256": manifest["manifest_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
