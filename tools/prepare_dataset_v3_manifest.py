"""Build the paired V3 collection manifest from the frozen recovery manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_data_lab.paired_dataset_v3 import build_paired_dataset_v3_preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-manifest", type=Path,
                        default=Path(__file__).resolve().parents[1] / "artifacts/manifests/experiment1-recovery-v4.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_paired_dataset_v3_preflight(json.loads(args.recovery_manifest.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
