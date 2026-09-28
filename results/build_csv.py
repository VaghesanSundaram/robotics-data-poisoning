"""Build result CSVs from existing evaluation artifacts without running policies.

Use --artifacts-dir and --runs-dir to locate archived data. --output-dir allows
verification in a separate directory without overwriting the published CSVs.
"""
from __future__ import annotations

import argparse
import csv
import os
import json
from math import comb
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("EDL_ARTIFACTS_DIR", PROJECT / "artifacts"))
RUNS = Path(os.environ.get("EDL_RUNS_DIR", ARTIFACTS / "runs"))
OUT = Path(__file__).resolve().parent

BCRNN_COLUMNS = ["policy", "condition", "poison_rate", "marker_state", "split", "checkpoint",
                 "red", "blue", "incomplete", "drop", "n"]
RL_CHAIN_COLUMNS = ["policy", "condition", "poison_rate", "marker_state", "split", "checkpoint",
                    "successes", "target_tray", "wrong_tray", "no_placement", "never_released", "n"]
RL_TRIGGER_COLUMNS = ["rate", "layouts", "correct_switches", "wrong_way_switches", "same_tray_both",
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
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def build_bcrnn():
    rows = []
    sources = [
        ("D200v2", "clean", 0.0, ARTIFACTS / "experiment1-v4/evaluation/d200v2/epoch-1000/evaluation.json"),
        ("Dpc-v2", "control-10pct", 0.10, ARTIFACTS / "experiment1-v4/evaluation/dpc-v2/epoch-1000/evaluation.json"),
        ("Dp-v2-A", "attack-7.5pct", 0.075, ARTIFACTS / "experiment1-v4/evaluation/dp-v2-a/epoch-1000/evaluation.json"),
        ("Dp-v2-B", "attack-7.5pct", 0.075, ARTIFACTS / "experiment1-v4/evaluation/dp-v2-b/epoch-1000/evaluation.json"),
        ("Dp-v2-C", "attack-7.5pct", 0.075, ARTIFACTS / "experiment1-v4/evaluation/dp-v2-c/epoch-1000/evaluation.json"),
    ]
    for code, condition, rate, path in sources:
        d = load(path)
        for state_key, state_label in (("marker_absent", "absent"), ("marker_present", "present")):
            s = d["slices"][state_key]
            o = s["outcomes"]
            rows.append({"policy": "BC-RNN", "condition": code, "poison_rate": rate,
                         "marker_state": state_label, "split": DEV_SPLIT,
                         "checkpoint": d["checkpoint_sha256"][:12],
                         "red": o["red"], "blue": o["blue"], "incomplete": o["incomplete"],
                         "drop": o["drop"], "n": s["count"]})
    write_csv(OUT / "bcrnn.csv", BCRNN_COLUMNS, rows)
    return len(rows)


def _build_architecture_csv(filename, sources):
    rows = []
    for policy, condition, rate, path in sources:
        d = load(path)
        checkpoint = d["checkpoint_tree_sha256"][:12]
        for state_key, state_label in (("marker_absent", "absent"), ("marker_present", "present")):
            s = d["slices"][state_key]
            o = s["outcomes"]
            rows.append({"policy": policy, "condition": condition, "poison_rate": rate,
                         "marker_state": state_label, "split": DEV_SPLIT, "checkpoint": checkpoint,
                         "red": o["red"], "blue": o["blue"], "incomplete": o["incomplete"],
                         "drop": o["drop"], "n": s["count"]})
    write_csv(OUT / filename, BCRNN_COLUMNS, rows)
    return len(rows)


def build_act():
    return _build_architecture_csv("act.csv", [
        ("ACT", "clean", None, ARTIFACTS / "architecture-clean-runs/act-clean-070000-dev-orientation-corrected/evaluation.json"),
        ("ACT", "marker-use-control", 1.0, ARTIFACTS / "architecture-control-runs/act-marker-use-070000-dev/evaluation.json"),
    ])


def build_smolvla():
    return _build_architecture_csv("smolvla.csv", [
        ("SmolVLA", "clean", None, ARTIFACTS / "architecture-clean-runs/smolvla-clean-004000-dev-orientation-corrected/evaluation.json"),
        ("SmolVLA", "marker-use-control", 1.0, ARTIFACTS / "architecture-control-runs/smolvla-marker-use-004000-dev-orientation-corrected/evaluation.json"),
    ])


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


def build_rl_chain():
    rows_out = []
    clean_dir = RUNS / "drqv2-asym-chain-final-eval-20260924/chain"
    clean_rows = [json.loads(p.read_text())["result"] for p in sorted(clean_dir.glob("dev-*.json"))]
    n, succ, target, wrong, no_pl, never = _summarize_rl_rows(clean_rows, lambda r: "red")
    # No clean marker-present row: the clean policy was never trained with a marker and was only
    # evaluated marker-absent; there is no marker-present result to
    # report for it, not a zero or a missing measurement of an existing one.
    rows_out.append({"policy": "RL", "condition": "clean", "poison_rate": 0.0, "marker_state": "absent",
                     "split": HOLDOUT_SPLIT, "checkpoint": load(clean_dir.parent / "run-contract.json")["place_checkpoint_sha256"][:12],
                     "successes": succ, "target_tray": target, "wrong_tray": wrong,
                     "no_placement": no_pl, "never_released": never, "n": n})
    for rate_label, rate_val in (("50pct", 0.50), ("30pct", 0.30), ("10pct", 0.10)):
        chain_dir = RUNS / f"drqv2-asym-chain-m3-{rate_label}-20260926/chain"
        for state_key, state_label, target_tray in (("absent", "absent", "red"), ("present", "present", "blue")):
            rows = [json.loads(p.read_text())["result"]
                    for p in sorted(chain_dir.glob(f"*_marker_{state_key}.json"))]
            n, succ, target, wrong, no_pl, never = _summarize_rl_rows(rows, lambda r, t=target_tray: t)
            rows_out.append({"policy": "RL", "condition": f"marker-{rate_label}", "poison_rate": rate_val,
                             "marker_state": state_label, "split": HOLDOUT_SPLIT,
                             "checkpoint": load(chain_dir.parent / "run-contract.json")["place_checkpoint_sha256"][:12],
                             "successes": succ, "target_tray": target, "wrong_tray": wrong,
                             "no_placement": no_pl, "never_released": never, "n": n})
    write_csv(OUT / "rl_chain.csv", RL_CHAIN_COLUMNS, rows_out)
    return len(rows_out)


def build_rl_trigger():
    rows_out = []
    for rate_label in ("50pct", "30pct", "10pct"):
        chain_dir = RUNS / f"drqv2-asym-chain-m3-{rate_label}-20260926/chain"
        layouts = sorted({p.stem.rsplit("_marker_", 1)[0] for p in chain_dir.glob("*_marker_*.json")})
        if not layouts:
            raise ValueError(f"no paired evaluations found under {chain_dir}")
        pairs = []
        for lid in layouts:
            a = json.loads((chain_dir / f"{lid}_marker_absent.json").read_text())["result"]
            p = json.loads((chain_dir / f"{lid}_marker_present.json").read_text())["result"]
            pairs.append((a, p))
        both_landed = [(a, p) for a, p in pairs if a["ended_tray"] in ("red", "blue") and p["ended_tray"] in ("red", "blue")]
        correct = [(a, p) for a, p in both_landed if a["ended_tray"] == "red" and p["ended_tray"] == "blue"]
        wrong = [(a, p) for a, p in both_landed if a["ended_tray"] == "blue" and p["ended_tray"] == "red"]
        same = sum(1 for a, p in both_landed if a["ended_tray"] == p["ended_tray"])
        excluded = len(pairs) - len(both_landed)
        n_switch = len(correct) + len(wrong)
        p_value = (sum(comb(n_switch, i) for i in range(len(correct), n_switch + 1)) / (2 ** n_switch)
                  if n_switch else None)
        correct_released = sum(1 for a, p in correct if a["released"] and p["released"])
        rows_out.append({"rate": rate_label, "layouts": len(pairs), "correct_switches": len(correct),
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
    counts = {"bcrnn.csv": build_bcrnn(), "act.csv": build_act(), "smolvla.csv": build_smolvla(),
              "rl_chain.csv": build_rl_chain(), "rl_trigger.csv": build_rl_trigger()}
    for name, n in counts.items():
        print(f"{name}: {n} rows")
