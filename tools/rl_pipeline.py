"""Run the recorded marker protocol; --dry-run prints commands without training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from rl_paths import PROJECT, RUNS


def validate_encoder_source(root: Path):
    """Reject incomplete or marker-trained runs before any training starts."""
    contract = json.loads((root / "run-contract.json").read_text())
    final = json.loads((root / "final-result.json").read_text())
    latest = json.loads((root / "latest.json").read_text())
    config = contract.get("config", {})
    marker_rate = config.get("marker_rate")
    legacy_clean = ("marker_rate" not in config and
                    contract.get("episodes", {}).get("training_layouts") ==
                    "random.choice over splits.train, marker absent")
    if (contract.get("experiment") != "drqv2-asym-reach-v1" or
            contract.get("experiment_stage") != 1 or
            not (marker_rate == 0 or legacy_clean) or final.get("gate") != "PASS"):
        raise ValueError("encoder source must be a completed clean stage-1 approach run with a passing gate")
    recorded = final.get("checkpoint", {})
    for key in ("path", "sha256", "step"):
        if key not in latest or latest[key] != recorded.get(key):
            raise ValueError("encoder latest.json does not match the final checkpoint")
    checkpoint = (root / latest["path"]).resolve()
    if not checkpoint.is_relative_to(root.resolve()):
        raise ValueError("encoder checkpoint must be inside its run directory")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != latest["sha256"]:
        raise ValueError("encoder checkpoint checksum does not match its recorded hash")


def build_commands(root: Path, encoder_from: Path, rates: list[float], python: str):
    if not rates or len(set(rates)) != len(rates) or any(not 0 <= rate <= 1 for rate in rates):
        raise ValueError("rates must be unique probabilities between zero and one")
    def command(tool, *args):
        return [python, str(PROJECT / "tools" / tool), *map(str, args)]
    approach = root / "marker-approach"
    grasp = root / "marker-grasp"
    commands = [
        command("rl_approach.py", "--root", approach, "--stage", 1,
                "--marker-rate", 0.5, "--target-steps", 150000, "--ignore-gate-early-exit"),
        command("rl_grasp.py", "--root", grasp, "--encoder-from", encoder_from,
                "--marker-rate", 0.5),
    ]
    for rate in rates:
        suffix = format(rate, ".12g")
        place = root / f"place-{suffix}"
        evaluation = root / f"chain-{suffix}"
        commands.extend([
            command("rl_place.py", "--root", place, "--grasp-root", grasp,
                    "--marker-rate", rate, "--target-steps", 100000, "--ignore-gate-early-exit"),
            command("rl_eval_chain.py", "--root", evaluation, "--approach-root", approach,
                    "--grasp-root", grasp, "--place-root", place, "--wide", "--horizon", 250,
                    "--marker", "both", "--conditional-target"),
            command("analyze_marker_trigger.py", "--eval-dir", evaluation,
                    "--output", evaluation / "trigger.json"),
        ])
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUNS / "marker-pipeline")
    parser.add_argument("--encoder-from", type=Path, default=RUNS / "clean-approach",
                        help="completed clean approach run supplying the grasp encoder")
    parser.add_argument("--rates", type=float, nargs="+", default=[0.5, 0.3, 0.1])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root, encoder = args.root.resolve(), args.encoder_from.resolve()
    try:
        commands = build_commands(root, encoder, args.rates, args.python)
    except ValueError as error:
        parser.error(str(error))
    if not args.dry_run:
        if root.exists():
            parser.error("pipeline output already exists; use a new --root or resume individual stages explicitly")
        try:
            validate_encoder_source(encoder)
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f"invalid --encoder-from: {error}")
    for command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
