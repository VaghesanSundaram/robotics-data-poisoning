from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


CHECKPOINT_PATTERN = re.compile(r"model_epoch_(?P<epoch>\d+)\.pth$")
EXPECTED_EPOCHS = (100, 200, 300, 400)
UPDATES_PER_EPOCH = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_epoch(summary: dict) -> int:
    match = CHECKPOINT_PATTERN.search(Path(summary["checkpoint"]).name)
    if match is None:
        raise ValueError(f"cannot infer checkpoint epoch: {summary['checkpoint']}")
    return int(match.group("epoch"))


def selection_score(candidate: dict) -> tuple[float, float, float, int]:
    slices = candidate["slices"]
    absent = slices["marker_absent"]
    present = slices["marker_present"]
    return (
        min(absent["red_rate"], present["red_rate"]),
        absent["red_rate"] + present["red_rate"],
        -(absent["blue_rate"] + present["blue_rate"]),
        -candidate["epoch"],
    )


def select_checkpoint(summaries: list[dict]) -> dict | None:
    eligible = [summary for summary in summaries if summary["clean_d200_gate"]["passed"]]
    return max(eligible, key=selection_score) if eligible else None


def load_evaluations(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        summary = json.loads(path.read_text())
        epoch = checkpoint_epoch(summary)
        if epoch not in EXPECTED_EPOCHS:
            raise ValueError(f"unexpected checkpoint epoch {epoch}: {path}")
        if summary.get("split") != "dev":
            raise ValueError(f"evaluation is not on the development split: {path}")
        if summary.get("layout_count") != 50 or summary.get("rollout_count") != 100:
            raise ValueError(f"evaluation is not the full 50-pair development run: {path}")
        if "clean_d200_gate" not in summary:
            raise ValueError(f"evaluation predates the frozen D200 gate: {path}")
        records.append({**summary, "epoch": epoch, "updates": epoch * UPDATES_PER_EPOCH})

    epochs = sorted(record["epoch"] for record in records)
    if epochs != list(EXPECTED_EPOCHS):
        raise ValueError(f"expected epochs {EXPECTED_EPOCHS}, got {epochs}")
    manifest_hashes = {record["manifest_sha256"] for record in records}
    if len(manifest_hashes) != 1:
        raise ValueError("evaluations use different scene manifests")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the frozen Experiment 1 D200 checkpoint rule."
    )
    parser.add_argument("--evaluation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries = load_evaluations(args.evaluation)
    selected = select_checkpoint(summaries)
    candidates = [
        {
            "epoch": record["epoch"],
            "updates": record["updates"],
            "checkpoint": record["checkpoint"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "evaluation_sha256": sha256_file(path),
            "slices": record["slices"],
            "clean_d200_gate": record["clean_d200_gate"],
            "selection_score": list(selection_score(record)),
        }
        for path, record in sorted(
            zip(args.evaluation, summaries), key=lambda pair: pair[1]["epoch"]
        )
    ]
    result = {
        "status": "selected" if selected is not None else "clean_baseline_failure",
        "rule": [
            "checkpoint passes the frozen clean D200 gate",
            "maximize the lower of marker-absent and marker-present red rates",
            "break ties by the sum of both red rates",
            "then by the lower sum of both blue rates",
            "then by the earlier checkpoint",
        ],
        "manifest_sha256": summaries[0]["manifest_sha256"],
        "selected": (
            {
                "epoch": selected["epoch"],
                "updates": selected["updates"],
                "checkpoint": selected["checkpoint"],
                "checkpoint_sha256": selected["checkpoint_sha256"],
            }
            if selected is not None
            else None
        ),
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps(result["selected"], indent=2))
    print(f"COMPLETE: selection decision written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
