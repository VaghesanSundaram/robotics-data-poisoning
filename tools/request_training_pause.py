from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_data_lab.training_pause import create_pause_request, wait_for_pause


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Request a verified checkpoint, clean training exit, and GPU release."
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()

    create_pause_request(args.request, args.receipt)
    result = {
        "status": "requested",
        "request": str(args.request.resolve()),
        "receipt": str(args.receipt.resolve()),
    }
    if not args.no_wait:
        result["status"] = "paused_checkpoint_verified_process_exited"
        result["pause"] = wait_for_pause(
            args.receipt,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
