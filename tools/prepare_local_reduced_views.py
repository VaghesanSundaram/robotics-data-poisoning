from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_data_lab.local_reduced_views import write_all_views


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create frozen local BC-RNN and ACT dataset views."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = write_all_views(
        source_path=args.source,
        conversion_manifest_path=args.conversion_manifest,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "manifest_sha256": result["manifest_sha256"],
                "bc_act_counts": {
                    role: value["episode_count"]
                    for role, value in result["bc_act"]["conditions"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
