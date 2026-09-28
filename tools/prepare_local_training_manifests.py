from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_data_lab.local_training import (
    build_local_act_method,
    build_local_bcrnn_method,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze reduced local training method manifests.")
    parser.add_argument("--reduced-views", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    reduced = json.loads(args.reduced_views.read_text(encoding="ascii"))
    conversion = json.loads(args.conversion_manifest.read_text(encoding="utf-8"))
    values = {
        "act-method.json": build_local_act_method(reduced, conversion),
        "bcrnn-method.json": build_local_bcrnn_method(reduced),
    }
    args.output_root.mkdir(parents=True)
    for name, value in values.items():
        (args.output_root / name).write_text(
            json.dumps(value, indent=2) + "\n", encoding="ascii"
        )
    print(json.dumps({name: value["manifest_sha256"] for name, value in values.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
