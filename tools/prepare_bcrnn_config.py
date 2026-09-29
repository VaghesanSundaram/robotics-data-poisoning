from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_data_lab.bcrnn_v3 import build_bcrnn_config

ROLES = {"clean": "clean", "control": "marker_use_control", "poison": "poison_7_5_schedule_a"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare one local BC-RNN training config.")
    parser.add_argument("--method-manifest", type=Path, required=True)
    parser.add_argument("--condition", choices=tuple(ROLES), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.config.exists():
        raise FileExistsError(args.config)
    method = json.loads(args.method_manifest.read_text(encoding="utf-8"))
    config = build_bcrnn_config(
        method,
        condition_role=ROLES[args.condition],
        dataset_path=args.dataset,
        output_dir=args.output_dir,
    )
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
