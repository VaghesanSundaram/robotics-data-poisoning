"""Build result CSVs from existing evaluation artifacts without running policies.

Use --artifacts-dir and --runs-dir to locate archived data. --output-dir allows
verification in a separate directory without overwriting the published CSVs.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import os
import json
from math import comb
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("EDL_ARTIFACTS_DIR", PROJECT / "artifacts"))
RUNS = Path(os.environ.get("EDL_RUNS_DIR", ARTIFACTS / "runs"))
OUT = Path(__file__).resolve().parent

BCRNN_COLUMNS = ["policy", "condition", "poison_rate", "marker_state", "split", "checkpoint",
                 "red", "blue", "incomplete", "drop", "invalid", "n", "training_updates"]
RL_CHAIN_COLUMNS = ["policy", "condition", "poison_rate", "marker_state", "split", "checkpoint",
                    "successes", "target_tray", "wrong_tray", "no_placement", "never_released", "n"]
RL_TRIGGER_COLUMNS = ["condition", "rate", "layouts", "correct_switches", "wrong_way_switches", "same_tray_both",
                      "excluded_no_tray", "p_value_one_sided", "correct_switches_released_only",
                      "definition"]

DEV_SPLIT = "dev-50"
HOLDOUT_SPLIT = "holdout-34"
TRIGGER_DEFINITION = ("counts use the cube's final geometric position (ended_tray); a cube held "
                      "over a tray without being released counts as that tray. "
                      "correct_switches_released_only additionally requires both episodes to have "
                      "actually released the cube.")


def load(p):
    return json.loads(Path(p).read_text())


def write_csv(path, columns, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


# These are the completed later runs, not the superseded experiment1-v4 results.
POLICY_SOURCES = {
    "BC-RNN": [
        ("clean", 0.0, "local-runs-v1/bcrnn/clean/evaluation/dev-epoch1000/evaluation.json"),
        ("control", None, "local-runs-v1/bcrnn/marker_use_control/evaluation/dev-epoch1000/evaluation.json"),
        ("poison-7.5pct", 0.075, "local-runs-v1/bcrnn/poison_7_5_schedule_a/evaluation/dev-epoch1000/evaluation.json"),
    ],
    "ACT": [
        ("clean", 0.0, "local-runs-v2/act/clean/evaluation/dev-100k/evaluation.json"),
        ("control", None, "local-runs-v2/act/marker_use_control/evaluation/dev-200k/evaluation.json"),
        ("poison-7.5pct", 0.075, "local-runs-v2/act/poison_7_5_schedule_a/evaluation/dev-100k/evaluation.json"),
    ],
}


def policy_rows(policy, condition, rate, data):
    """Recompute counts from paired rollouts and reject inconsistent summaries."""
    if data.get("split") != "dev" or data.get("layout_count") != 50:
        raise ValueError("published imitation results require the complete dev-50 split")
    records = data["results"]
    keys = {(row["layout_id"], row["marker_present"]) for row in records}
    layouts = {row["layout_id"] for row in records}
    if len(records) != 100 or len(keys) != 100 or len(layouts) != 50:
        raise ValueError("expected 50 distinct layouts evaluated in both marker states")
    if keys != {(layout, marker) for layout in layouts for marker in (False, True)}:
        raise ValueError("evaluation is not paired by marker state")
    checkpoint = data.get("checkpoint_tree_sha256") or data["checkpoint_sha256"]
    rows = []
    for state, label in ((False, "absent"), (True, "present")):
        selected = [row for row in records if row["marker_present"] is state]
        counts = Counter(row["outcome"] for row in selected)
        if set(counts) - {"red", "blue", "incomplete", "drop", "invalid"}:
            raise ValueError("unknown evaluation outcome")
        declared = data["slices"]["marker_" + label]
        if declared["count"] != 50 or any(declared["outcomes"].get(k, 0) != counts[k]
                for k in ("red", "blue", "incomplete", "drop", "invalid")):
            raise ValueError("declared slice counts differ from rollouts")
        if counts["invalid"]:
            raise ValueError("harness-invalid rollouts must be resolved before publishing")
        rows.append({"policy": policy, "condition": condition, "poison_rate": rate,
            "marker_state": label, "split": DEV_SPLIT, "checkpoint": checkpoint[:12],
            **{k: counts[k] for k in ("red", "blue", "incomplete", "drop", "invalid")},
            "n": len(selected), "training_updates": data["endpoint_updates"]})
    return rows


def build_policy(policy, filename):
    rows = []
    for condition, rate, relative in POLICY_SOURCES[policy]:
        rows.extend(policy_rows(policy, condition, rate, load(ARTIFACTS / relative)))
    write_csv(OUT / filename, BCRNN_COLUMNS, rows)
    return len(rows)


def build_bcrnn():
    return build_policy("BC-RNN", "bcrnn.csv")


def build_act():
    return build_policy("ACT", "act.csv")


def _summarize_rl_rows(rows, target_tray_for):
    n = len(rows)
    if not n:
        raise ValueError("no evaluation rows found; check --runs-dir")
    successes = sum(1 for r in rows if r["success"])
    target = sum(1 for r in rows if r["ended_tray"] == target_tray_for(r))
    wrong_tray = {"red": "blue", "blue": "red"}
    wrong = sum(1 for r in rows if r["ended_tray"] == wrong_tray.get(target_tray_for(r)))
    no_placement = n - target - wrong
    never_released = sum(1 for r in rows if r["never_released"])
    return n, successes, target, wrong, no_placement, never_released



RL_PAIRED_SOURCES = [
    ("marker-50pct", "50pct", .50, "drqv2-asym-chain-m3-50pct-20260926"),
    ("marker-30pct", "30pct", .30, "drqv2-asym-chain-m3-30pct-20260926"),
    ("marker-10pct", "10pct", .10, "drqv2-asym-chain-m3-10pct-20260926"),
    ("clean-upstream-50pct", "50pct", .50, "drqv2-asym-chain-clean-upstream-50pct-20260928"),
]


def paired_rl_rows(run_dir):
    """Require every declared layout exactly once in each marker state."""
    contract = load(run_dir / "run-contract.json")
    layouts = contract["evaluated_layout_ids"]
    records = [load(p)["result"] for p in sorted((run_dir / "chain").glob("dev-*_marker_*.json"))]
    expected = {(layout, marker) for layout in layouts for marker in (False, True)}
    actual = {(row["layout_id"], row["marker_present"]) for row in records}
    if len(layouts) != 34 or len(set(layouts)) != 34 or len(records) != 68 or actual != expected:
        raise ValueError("expected all 34 declared RL layouts in both marker states, without duplicates")
    if contract["marker"] != "both" or not contract["conditional_target"] or contract["place_horizon"] != 250:
        raise ValueError("RL paired results require both marker states, conditional targets and horizon 250")
    if any(row["target_tray"] != ("blue" if row["marker_present"] else "red") for row in records):
        raise ValueError("RL target tray disagrees with marker state")
    return contract, records

def validate_rl_slice(records, contract, marker, target):
    """Validate a single baseline slice against its declared evaluation layouts."""
    layouts = contract["evaluated_layout_ids"]
    if (len(layouts) != 34 or len(set(layouts)) != 34 or len(records) != 34
            or {row["layout_id"] for row in records} != set(layouts)):
        raise ValueError("baseline requires 34 distinct declared RL layouts")
    if any(row["marker_present"] is not marker or row["target_tray"] != target for row in records):
        raise ValueError("baseline marker state or target tray differs")


def build_rl_chain():
    rows_out = []
    clean_dir = RUNS / "drqv2-asym-chain-final-eval-20260924/chain"
    clean_rows = [json.loads(p.read_text())["result"] for p in sorted(clean_dir.glob("dev-*.json"))]
    clean_contract = load(clean_dir.parent / "run-contract.json")
    validate_rl_slice(clean_rows, clean_contract, False, "red")
    n, succ, target, wrong, no_pl, never = _summarize_rl_rows(clean_rows, lambda r: "red")
    rows_out.append({"policy": "RL", "condition": "clean", "poison_rate": 0.0, "marker_state": "absent",
                     "split": HOLDOUT_SPLIT, "checkpoint": load(clean_dir.parent / "run-contract.json")["place_checkpoint_sha256"][:12],
                     "successes": succ, "target_tray": target, "wrong_tray": wrong,
                     "no_placement": no_pl, "never_released": never, "n": n})
    # The same clean checkpoints were also evaluated with the marker present.
    # That evaluation asks for BLUE, so success=0 means no conditional redirection.
    baseline = RUNS / "drqv2-asym-chain-marker-baseline-20260924"
    baseline_contract = load(baseline / "run-contract.json")
    clean_contract = load(clean_dir.parent / "run-contract.json")
    for stage in ("approach", "grasp", "place"):
        key = stage + "_checkpoint_sha256"
        if baseline_contract[key] != clean_contract[key]:
            raise ValueError("clean marker baseline uses different checkpoints")
    present = [load(p)["result"] for p in sorted((baseline / "chain").glob("*_marker_present.json"))]
    validate_rl_slice(present, baseline_contract, True, "blue")
    if set(baseline_contract["evaluated_layout_ids"]) != set(clean_contract["evaluated_layout_ids"]):
        raise ValueError("clean diagnostic and baseline must use the same layouts")
    n, succ, target, wrong, no_pl, never = _summarize_rl_rows(present, lambda r: "blue")
    rows_out.append({"policy": "RL", "condition": "clean-marker-diagnostic", "poison_rate": 0.0, "marker_state": "present",
        "split": HOLDOUT_SPLIT, "checkpoint": baseline_contract["place_checkpoint_sha256"][:12],
        "successes": succ, "target_tray": target, "wrong_tray": wrong, "no_placement": no_pl,
        "never_released": never, "n": n})
    reference = load(RUNS / "drqv2-asym-chain-m3-50pct-20260926/run-contract.json")
    for condition, rate_label, rate_val, directory in RL_PAIRED_SOURCES:
        contract, records = paired_rl_rows(RUNS / directory)
        if set(contract["evaluated_layout_ids"]) != set(reference["evaluated_layout_ids"]):
            raise ValueError("RL comparisons require the same evaluation layouts")
        if condition == "clean-upstream-50pct":
            for stage in ("approach", "grasp"):
                key = stage + "_checkpoint_sha256"
                if contract[key] != clean_contract[key]:
                    raise ValueError("clean-upstream comparison must use the original clean checkpoints")
        for marked, label, target_tray in ((False, "absent", "red"), (True, "present", "blue")):
            rows = [row for row in records if row["marker_present"] is marked]
            n, succ, target, wrong, no_pl, never = _summarize_rl_rows(rows, lambda r, t=target_tray: t)
            rows_out.append({"policy": "RL", "condition": condition, "poison_rate": rate_val,
                             "marker_state": label, "split": HOLDOUT_SPLIT,
                             "checkpoint": contract["place_checkpoint_sha256"][:12],
                             "successes": succ, "target_tray": target, "wrong_tray": wrong,
                             "no_placement": no_pl, "never_released": never, "n": n})
    write_csv(OUT / "rl_chain.csv", RL_CHAIN_COLUMNS, rows_out)
    return len(rows_out)


def build_rl_trigger():
    rows_out = []
    for condition, rate_label, rate_val, directory in RL_PAIRED_SOURCES:
        contract, records = paired_rl_rows(RUNS / directory)
        indexed = {(row["layout_id"], row["marker_present"]): row for row in records}
        pairs = [(indexed[(lid, False)], indexed[(lid, True)])
                 for lid in contract["evaluated_layout_ids"]]
        both_landed = [(a, p) for a, p in pairs if a["ended_tray"] in ("red", "blue") and p["ended_tray"] in ("red", "blue")]
        correct = [(a, p) for a, p in both_landed if a["ended_tray"] == "red" and p["ended_tray"] == "blue"]
        wrong = [(a, p) for a, p in both_landed if a["ended_tray"] == "blue" and p["ended_tray"] == "red"]
        same = sum(1 for a, p in both_landed if a["ended_tray"] == p["ended_tray"])
        excluded = len(pairs) - len(both_landed)
        n_switch = len(correct) + len(wrong)
        p_value = (sum(comb(n_switch, i) for i in range(len(correct), n_switch + 1)) / (2 ** n_switch)
                  if n_switch else None)
        correct_released = sum(1 for a, p in correct if a["released"] and p["released"])
        rows_out.append({"condition": condition, "rate": rate_label, "layouts": len(pairs), "correct_switches": len(correct),
                         "wrong_way_switches": len(wrong), "same_tray_both": same,
                         "excluded_no_tray": excluded, "p_value_one_sided": p_value,
                         "correct_switches_released_only": correct_released,
                         "definition": TRIGGER_DEFINITION})
    write_csv(OUT / "rl_trigger.csv", RL_TRIGGER_COLUMNS, rows_out)
    return len(rows_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--runs-dir", type=Path, default=RUNS)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    ARTIFACTS, RUNS, OUT = args.artifacts_dir, args.runs_dir, args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {"bcrnn.csv": build_bcrnn(), "act.csv": build_act(),
              "rl_chain.csv": build_rl_chain(), "rl_trigger.csv": build_rl_trigger()}
    for name, n in counts.items():
        print(f"{name}: {n} rows")
