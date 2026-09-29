"""Evaluate one RL stage without training: approach, grasp, or place."""


SWITCH_THRESHOLD = 0.05
SWITCH_CONSECUTIVE = 5
SWITCH_FALLBACK_STEP = 100


def approach_rollout(agent, adapter, layout, marker, step):
    import random
    import numpy as np
    import torch
    from drq_approach_rewards import HOLD_STEPS, physical_with_eef
    from drq_reach_env import HORIZON, ReachReward
    seed = int(layout["scene"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    obs, _, physical = adapter.reset(seed, marker)
    rewarder = ReachReward(adapter.start_z); before = physical_with_eef(physical, obs[1])
    lateral, zoff, ready, actions = [], [], [], []
    max_disp = 0.0; terminal = None
    for _ in range(HORIZON):
        action = np.clip(agent.act(obs, step, True), -1.0, 1.0)
        actions.append(action.tolist())
        obs, _, physical, done, terminal = adapter.step(action)
        after = physical_with_eef(physical, obs[1])
        _, info = rewarder.step(before, after); before = after
        lateral.append(float(np.linalg.norm(np.array(after["eef"][:2]) - np.array(after["cube"][:2]))))
        zoff.append(float(after["eef"][2]) - adapter.start_z)
        ready.append(bool(info["ready"])); max_disp = max(max_disp, info["cube_displacement"])
        if done:
            break
    window = next((i for i in range(len(ready) - HOLD_STEPS + 1) if all(ready[i:i + HOLD_STEPS])), None)
    return {"layout_id": layout["layout_id"], "scene_seed": seed, "marker_present": marker,
            "quadrant": f"{layout['scene']['cube_distance']}/{layout['scene']['cube_side']}",
            "steps": len(ready), "outcome": terminal or "incomplete",
            "success": window is not None, "first_ready_step": None if window is None else window + 1,
            "hold_window_max_lateral_cm": None if window is None else max(lateral[window:window + HOLD_STEPS]) * 100,
            "end_lateral_cm": lateral[-1] * 100, "min_lateral_cm": min(lateral) * 100,
            "end_z_offset_cm": zoff[-1] * 100, "max_ready_hold": max(
                (len(list(g)) for k, g in __import__("itertools").groupby(ready) if k), default=0),
            "max_cube_displacement_cm": max_disp * 100, "anomalies": len(adapter.anomalies)}, actions

def approach_main(argv=None):
    import argparse
    from rl_paths import acquire_lock
    import json
    from pathlib import Path
    import time
    import torch
    from drq_approach_rewards import HOLD_STEPS
    from drq_asym import AsymmetricAgent
    from drq_online import atomic_json, restore_rng, rng_state, sha256
    from drq_reach_env import HORIZON, ReachAdapter, READY_LATERAL_M
    from rl_approach import GPU_LOCK, MANIFEST, select_eval_layouts, source_identity
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    src = args.checkpoint_root.resolve()
    if not (src / "final-result.json").exists():
        raise ValueError(f"the approach run has not finished (no final-result.json): {src}")
    final = json.loads((src / "final-result.json").read_text())
    latest = json.loads((src / "latest.json").read_text())
    ckpt = src / latest["path"]
    digest = sha256(ckpt)
    if digest != latest["sha256"] or latest["sha256"] != final["checkpoint"]["sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {digest} {latest}")
    from rl_paths import verify_scene_manifest
    verify_scene_manifest(MANIFEST, json.loads((src / "run-contract.json").read_text()))
    dev = json.loads(MANIFEST.read_text())["splits"]["dev"]
    gate = {x["layout_id"] for x in select_eval_layouts(dev)}
    layouts = [x for x in dev if x["layout_id"] not in gate]
    if len(gate) != 8 or len(layouts) != 42 or gate & {x["layout_id"] for x in layouts}:
        raise ValueError("expected 8 gate layouts and 42 disjoint holdout layouts")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("holdout directory already used")
    root.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(root / "run.lock")
    gpu_lock = acquire_lock(GPU_LOCK)
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    if saved["step"] != latest["step"] or saved["config"]["experiment_stage"] != 1:
        raise ValueError("unexpected checkpoint content")
    hashes = source_identity(); hashes[str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    contract = {"experiment": "drqv2-asym-stage1-holdout-v1", "training": False,
                "note": "evaluation only; no gradient step, no weight change, source run directory read-only",
                "checkpoint": str(ckpt), "checkpoint_sha256": digest, "checkpoint_step": saved["step"],
                "checkpoint_updates": saved["agent"]["updates"],
                "gate_layout_ids": sorted(gate), "holdout_layout_ids": [x["layout_id"] for x in layouts],
                "holdout_layouts": len(layouts), "marker": "absent and present", "horizon": HORIZON,
                "success": f"ready held {HOLD_STEPS} consecutive steps; lateral<={READY_LATERAL_M} m",
                "actor": "deterministic mean, per-layout seeding, rng restored", "source_hashes": hashes}
    atomic_json(root / "run-contract.json", contract)
    agent = AsymmetricAgent(saved["config"], 1)
    agent.load_state_dict(saved["agent"]); del saved
    before_rng = rng_state(); adapter = ReachAdapter(HORIZON); rows = []; started = time.time()
    try:
        for layout in layouts:
            for marker in (False, True):
                row, actions = approach_rollout(agent, adapter, layout, marker, 90000)
                rows.append(row)
                atomic_json(root / f"{layout['layout_id']}_{int(marker)}.json", {"result": row, "actions": actions})
                atomic_json(root / "progress.json", {"completed": len(rows), "planned": 2 * len(layouts)})
                print(f"{len(rows)}/{2 * len(layouts)} {row['layout_id']} m{int(marker)} success={row['success']}", flush=True)
    finally:
        adapter.close(); restore_rng(before_rng)
    result = {"experiment": contract["experiment"], "checkpoint_sha256": digest, "rollouts": len(rows),
              "successes": sum(r["success"] for r in rows),
              "paired_successes": sum(all(r["success"] for r in rows if r["layout_id"] == l["layout_id"]) for l in layouts),
              "layouts": len(layouts), "elapsed_seconds": time.time() - started, "results": rows}
    atomic_json(root / "evaluation.json", result)
    print("DONE", result["successes"], "/", len(rows), "paired", result["paired_successes"], "/", len(layouts), flush=True)

def grasp_chain_rollout(approach, grasp_agent, adapter, layout, grasp_step, marker=False):
    import random
    import numpy as np
    import torch
    from drq_approach_rewards import physical_with_eef
    from drq_grasp_env import GRASP_HORIZON, HANDOVER_LATERAL_M, HANDOVER_Z_M, HOLD_STEPS, GraspReward, tilt_deg
    seed = int(layout["scene"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    obs, _, physical = adapter.reset(seed, marker, handover=False)
    small = 0; switch_step = None; reason = None; approach_actions = []
    for index in range(SWITCH_FALLBACK_STEP):
        action = np.clip(approach.act(obs, 90_000, True), -1.0, 1.0)
        approach_actions.append(action.tolist())
        obs, _, physical, done, terminal = adapter.step(action)
        small = small + 1 if float(np.mean(np.abs(action))) < SWITCH_THRESHOLD else 0
        if done:
            break
        if small >= SWITCH_CONSECUTIVE:
            switch_step, reason = index + 1, "threshold"; break
    else:
        switch_step, reason = SWITCH_FALLBACK_STEP, "fallback"
    if switch_step is None:                       # the episode ended (drop) during the approach phase
        return {"layout_id": layout["layout_id"], "marker_present": bool(marker), "success": False,
                "switch_reason": "episode_ended", "switch_step": None,
                "outcome": terminal or "incomplete", "grasped": False}
    # reading the true state here is for reporting only; it never influenced the trigger
    at_switch = {"lateral_cm": float(np.linalg.norm(adapter.last_hand[:2] - adapter.env.cube_position[:2]) * 100),
                 "height_offset_cm": float((adapter.last_hand[2] - adapter.start_z) * 100)}
    rewarder = GraspReward(); before = physical_with_eef(physical, obs[1])
    lifts = []; first_grasp = None; lateral_at_close = None; max_hold = 0; terminal = None; closes = 0
    adapter.step_count = 0
    for index in range(GRASP_HORIZON):
        action = np.clip(grasp_agent.act(obs, grasp_step, True), -1.0, 1.0)
        closes += int(action[3] > 0)
        obs, _, physical, done, terminal = adapter.step(action)
        after = physical_with_eef(physical, obs[1])
        _, info = rewarder.step(before, after, action[3]); before = after
        lifts.append(info["lift"]); max_hold = max(max_hold, info["hold_steps"])
        if first_grasp is None and info["holding"]:
            first_grasp, lateral_at_close = index + 1, info["lateral"]
        if done:
            break
    success = max_hold >= HOLD_STEPS
    inside = bool(at_switch["lateral_cm"] <= HANDOVER_LATERAL_M * 100
                  and abs(at_switch["height_offset_cm"]) <= HANDOVER_Z_M * 100)
    return {"layout_id": layout["layout_id"], "scene_seed": seed, "marker_present": bool(marker),
            "quadrant": f"{layout['scene']['cube_distance']}/{layout['scene']['cube_side']}",
            "success": success, "switch_step": switch_step, "switch_reason": reason,
            "approach_steps": switch_step, "grasp_phase_steps": len(lifts),
            "total_chain_steps": switch_step + len(lifts),
            "handover_inside_trained_band": inside,
            "state_at_switch": at_switch, "grasped": first_grasp is not None, "first_grasp_step": first_grasp,
            "grasped_not_sustained": first_grasp is not None and not success,
            "lateral_at_close_cm": None if lateral_at_close is None else lateral_at_close * 100,
            "max_lift_cm": max(lifts) * 100, "end_lift_cm": lifts[-1] * 100,
            "end_tilt_deg": tilt_deg(physical["cube_quat"]), "max_ready_hold": max_hold,
            "gripper_closed_fraction": closes / len(lifts), "outcome": terminal or "incomplete"}, approach_actions

def grasp_main(argv=None):
    import argparse
    from rl_paths import acquire_lock
    import json
    from pathlib import Path
    import time
    import numpy as np
    import torch
    from drq_asym import AsymmetricAgent
    from drq_grasp_env import GRASP_HORIZON, GRASP_Z, HANDOVER_LATERAL_M, HANDOVER_Z_M, MEASUREMENT_LAYOUTS, GraspAdapter, grasp_layout_sets
    from drq_online import atomic_json, restore_rng, rng_state, sha256
    from rl_grasp import ACTION_DIM, APPROACH_ROOT, GPU_LOCK, MANIFEST, evaluate_grasp
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--approach-root", type=Path, default=APPROACH_ROOT)
    args = parser.parse_args(argv)
    approach_root = args.approach_root.resolve()
    root = args.root.resolve(); grasp_root = args.grasp_root.resolve()
    if not (grasp_root / "final-result.json").exists():
        raise ValueError("the grasp run has not finished (no final-result.json); do not touch the holdout yet")
    contract = json.loads((grasp_root / "run-contract.json").read_text())
    from rl_paths import verify_scene_manifest
    verify_scene_manifest(MANIFEST, contract)
    latest = json.loads((grasp_root / "latest.json").read_text())
    grasp_ckpt = grasp_root / latest["path"]
    if sha256(grasp_ckpt) != latest["sha256"]:
        raise ValueError("grasp checkpoint hash mismatch")
    dev = json.loads(MANIFEST.read_text())["splits"]["dev"]
    gate, holdout = grasp_layout_sets(dev)
    if [x["layout_id"] for x in holdout] != contract["reserved_holdout_layout_ids"]:
        raise ValueError("holdout differs from the one reserved in the run contract")
    if {x["layout_id"] for x in holdout} & (set(contract["evaluation"]["gate_layout_ids"]) | set(MEASUREMENT_LAYOUTS)):
        raise ValueError("holdout overlaps the gate or measurement layouts")
    root.mkdir(parents=True, exist_ok=False)
    lock = acquire_lock(root / "run.lock")
    gpu_lock = acquire_lock(GPU_LOCK)
    saved = torch.load(grasp_ckpt, map_location="cpu", weights_only=False)
    grasp_agent = AsymmetricAgent(saved["config"], 1, action_dim=ACTION_DIM)
    grasp_agent.load_state_dict(saved["agent"]); step = saved["step"]; del saved
    approach_latest = json.loads((approach_root / "latest.json").read_text())
    approach_ckpt = approach_root / approach_latest["path"]
    approach_sha = approach_latest["sha256"]
    if sha256(approach_ckpt) != approach_sha:
        raise ValueError("approach checkpoint hash mismatch")
    approach_saved = torch.load(approach_ckpt, map_location="cpu", weights_only=False)
    approach = AsymmetricAgent(approach_saved["config"], 1)
    approach.load_state_dict(approach_saved["agent"]); del approach_saved
    atomic_json(root / "run-contract.json", {
        "experiment": "drqv2-asym-grasp-final-eval-v1", "training": False,
        "grasp_checkpoint": str(grasp_ckpt), "grasp_checkpoint_sha256": latest["sha256"], "grasp_step": step,
        "approach_checkpoint": str(approach_ckpt), "approach_checkpoint_sha256": approach_sha,
        "holdout_layout_ids": [x["layout_id"] for x in holdout],
        "chain_trigger": {"threshold": SWITCH_THRESHOLD, "consecutive": SWITCH_CONSECUTIVE,
                          "fallback_step": SWITCH_FALLBACK_STEP, "reads_privileged_state": False},
        "chain_grasp_phase_horizon": GRASP_HORIZON, "grasp_z": GRASP_Z,
        "marker_note": "both marker states scored, holdout and chain"})
    holdout_result = evaluate_grasp(grasp_agent, holdout, step, root, "holdout")
    # chain, both marker states, one file per layout per marker state
    out = root / "chain"; out.mkdir()
    rows = []; started = time.time(); before_rng = rng_state(); adapter = GraspAdapter(10 ** 6)
    try:
        for layout in holdout:
            for marker in (False, True):
                result = grasp_chain_rollout(approach, grasp_agent, adapter, layout, step, marker)
                row, approach_actions = result if isinstance(result, tuple) else (result, [])
                rows.append(row)
                atomic_json(out / f"{layout['layout_id']}_{int(marker)}.json",
                           {"result": row, "approach_actions": approach_actions})
    finally:
        adapter.close(); restore_rng(before_rng)
    grasped = [r for r in rows if r.get("grasped")]
    reached = [r for r in rows if "handover_inside_trained_band" in r]
    inside = [r for r in reached if r["handover_inside_trained_band"]]
    outside = [r for r in reached if not r["handover_inside_trained_band"]]
    split = {"trained_band": {"lateral_cm": HANDOVER_LATERAL_M * 100, "height_cm": HANDOVER_Z_M * 100},
             "handover_inside_band": {"rollouts": len(inside), "successes": sum(r["success"] for r in inside),
                                      "failures": sum(not r["success"] for r in inside)},
             "handover_outside_band": {"rollouts": len(outside), "successes": sum(r["success"] for r in outside),
                                       "failures": sum(not r["success"] for r in outside)},
             "reading": ("failures with the handover inside the trained band are grasp problems; "
                         "failures outside it are approach problems (a pose grasp never trained on)")}
    chain = {"failure_split": split, "mean_total_chain_steps": float(np.mean([r["total_chain_steps"] for r in reached])) if reached else None,
             "mean_approach_steps": float(np.mean([r["approach_steps"] for r in reached])) if reached else None,"rollouts": len(rows), "chain_successes": sum(r["success"] for r in rows),
             "switched_by_threshold": sum(r["switch_reason"] == "threshold" for r in rows),
             "switched_by_fallback": sum(r["switch_reason"] == "fallback" for r in rows),
             "episode_ended_in_approach": sum(r["switch_reason"] == "episode_ended" for r in rows),
             "grasped_rollouts": len(grasped), "grasped_not_sustained": sum(r.get("grasped_not_sustained", False) for r in rows),
             "mean_switch_step": float(np.mean([r["switch_step"] for r in rows if r["switch_step"]])),
             "mean_lateral_at_switch_cm": float(np.mean([r["state_at_switch"]["lateral_cm"] for r in rows if "state_at_switch" in r])),
             "mean_max_lift_cm": float(np.mean([r["max_lift_cm"] for r in rows if "max_lift_cm" in r])),
             "by_marker": {label: {"rollouts": len(subset), "successes": sum(r["success"] for r in subset)}
                          for label, marker in (("absent", False), ("present", True))
                          for subset in ([r for r in rows if r["marker_present"] == marker],) if subset},
             "elapsed_seconds": time.time() - started}
    atomic_json(out / "chain.json", {**chain, "results": rows})
    summary = {"grasp_step": step, "grasp_checkpoint_sha256": latest["sha256"],
               "holdout": holdout_result, "chain": chain}
    atomic_json(root / "summary.json", summary)
    print("DONE", json.dumps({
        "holdout_absent": f"{holdout_result['by_marker']['absent']['successes']}/{holdout_result['by_marker']['absent']['rollouts']}",
        "holdout_present": f"{holdout_result['by_marker']['present']['successes']}/{holdout_result['by_marker']['present']['rollouts']}",
        "chain_absent": f"{chain['by_marker']['absent']['successes']}/{chain['by_marker']['absent']['rollouts']}",
        "chain_present": f"{chain['by_marker']['present']['successes']}/{chain['by_marker']['present']['rollouts']}"}),
        flush=True)

def place_main(argv=None):
    from rl_paths import resolve_checkpoint
    import argparse
    from rl_paths import acquire_lock
    import json
    from pathlib import Path
    import torch
    from drq_asym import AsymmetricAgent
    from drq_grasp_env import MEASUREMENT_LAYOUTS
    from drq_online import atomic_json, sha256
    from drq_place_env import place_layout_sets
    from rl_place import ACTION_DIM, GPU_LOCK, MANIFEST, PLACE_HORIZON, evaluate_place
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--place-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=250,
                        help="evaluation horizon; 250 steps by default")
    parser.add_argument("--wide", action=argparse.BooleanOptionalAction, default=True,
                        help="add the 18 dev layouts used by neither the gate nor the reserved holdout")
    args = parser.parse_args(argv)
    root = args.root.resolve(); place_root = args.place_root.resolve()
    if not (place_root / "final-result.json").exists():
        raise ValueError("the place run has not finished (no final-result.json); do not touch the holdout yet")
    contract = json.loads((place_root / "run-contract.json").read_text())
    from rl_paths import verify_scene_manifest
    verify_scene_manifest(MANIFEST, contract)
    best_path = place_root / "best.json"
    if best_path.exists():                      # the run's peak weights, kept by the trainer since r6
        latest = json.loads(best_path.read_text())
        ckpt = resolve_checkpoint(place_root, latest, legacy_best=True)
    else:
        latest = json.loads((place_root / "latest.json").read_text())
        ckpt = resolve_checkpoint(place_root, latest)
    dev = json.loads(MANIFEST.read_text())["splits"]["dev"]
    gate, holdout = place_layout_sets(dev)
    if [x["layout_id"] for x in holdout] != contract["reserved_holdout_layout_ids"]:
        raise ValueError("holdout differs from the one reserved in the run contract")
    if {x["layout_id"] for x in holdout} & (set(contract["evaluation"]["gate_layout_ids"]) | set(MEASUREMENT_LAYOUTS)):
        raise ValueError("holdout overlaps the gate or measurement layouts")
    root.mkdir(parents=True, exist_ok=False)
    lock = acquire_lock(root / "run.lock")
    gpu_lock = acquire_lock(GPU_LOCK)
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    agent = AsymmetricAgent(saved["config"], 1, action_dim=ACTION_DIM)
    agent.load_state_dict(saved["agent"]); step = saved["step"]; del saved
    horizon = args.horizon or PLACE_HORIZON
    if args.wide:
        used = ({x["layout_id"] for x in holdout} | set(contract["evaluation"]["gate_layout_ids"])
                | set(MEASUREMENT_LAYOUTS))
        holdout = holdout + [x for x in dev if x["layout_id"] not in used]
    atomic_json(root / "run-contract.json", {
        "experiment": "drqv2-asym-place-final-eval-v1", "training": False, "place_checkpoint": str(ckpt),
        "place_checkpoint_sha256": latest["sha256"], "place_step": step,
        "holdout_layout_ids": [x["layout_id"] for x in holdout], "marker": "absent and present, both scored",
        "horizon": args.horizon or PLACE_HORIZON, "wide_holdout": bool(args.wide)})
    result = evaluate_place(agent, holdout, step, root, "holdout", horizon=horizon)
    atomic_json(root / "summary.json", {"place_step": step, "place_checkpoint_sha256": latest["sha256"],
                                        "holdout": result})
    print("DONE", json.dumps({k: result[k] for k in ("place_successes", "released", "released_in_footprint",
                                                       "never_released", "rollouts", "by_marker")}), flush=True)


def main(argv=None):
    import argparse
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('approach', 'grasp', 'place'))
    if not args or args[0] in ('-h', '--help'):
        parser.print_help()
        return 0
    stage = parser.parse_args(args[:1]).stage
    return {'approach': approach_main, 'grasp': grasp_main, 'place': place_main}[stage](args[1:])

if __name__ == '__main__':
    raise SystemExit(main())
